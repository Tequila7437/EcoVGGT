"""Online Stream: EcoVGGT on frames that arrive one at a time.

Every `key_interval` frames, the newly arrived frame becomes a candidate key. A pass then
pins the candidate keys (thinned evenly in time to at most `max_keys`), rebuilds the Scene
Cache, attaches the frames whose turn has come, and emits them. Each frame is emitted
exactly once, `lag` passes after the interval it belongs to, so it is reconstructed with
keys that reach beyond it. The scale of every pass is stitched to the scale already
emitted through the median depth of the keys shared with earlier passes (VGGT), or with a
similarity transform fitted on the shared keys (pi3).

Usage:
    stream = VGGTOnlineStream(model, key_interval=16, lag=2, max_keys=32)
    for image in frames:              # image: [3, H, W] in [0, 1]
        for i in stream.push(image):  # indices of frames emitted by this call
            ...                       # stream.pose_enc(i), stream.depth[i], stream.conf[i]
    stream.finish()                   # emits the remaining frames
"""
import contextlib
from typing import Dict, List

import numpy as np
import torch

from . import pi3_engine
from .geometry import umeyama_alignment
from .vggt_engine import attach_pass, head_inputs, pin_pass


def _thin(candidates: List[int], m: int, max_keys: int) -> List[int]:
    """Key set of pass m: all candidate keys, or `max_keys` of them spread evenly in time
    (always including frame 0 and the newest candidate)."""
    cand = candidates[:m + 1]
    if not max_keys or len(cand) <= max_keys:
        return cand
    return [cand[i] for i in np.round(np.linspace(0, m, max_keys)).astype(int)]


class _Buffer:
    """Arrived frames, kept until they are emitted and can no longer be pinned."""

    def __init__(self, device):
        self.frames: Dict[int, torch.Tensor] = {}
        self.device = device

    def add(self, i, image):
        self.frames[i] = image

    def stack(self, idx):
        return torch.stack([self.frames[i] for i in idx], 0)[None].to(self.device)

    def release(self, emitted, candidates):
        keep = set(candidates)
        for i in [i for i in self.frames if i in emitted and i not in keep]:
            del self.frames[i]


class VGGTOnlineStream:
    """Online Stream on a VGGT backbone. Outputs are pose encodings (VGGT convention, frame 0
    is the reference), depth maps and their confidence, all in the scale of the first pass."""

    def __init__(self, model, key_interval=16, lag=2, max_keys=32, chunk=64, tasks=("camera", "depth"), amp_dtype=torch.bfloat16):
        self.model, self.kappa, self.lag, self.max_keys, self.chunk, self.tasks = model, key_interval, lag, max_keys, chunk, tasks
        self.dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device
        self.amp_dtype = amp_dtype
        self.buf = _Buffer(self.device)
        self.n = 0
        self.candidates: List[int] = []
        self.pose: Dict[int, torch.Tensor] = {}
        self.depth: Dict[int, torch.Tensor] = {}
        self.conf: Dict[int, torch.Tensor] = {}
        self.med: Dict[int, float] = {}
        self.scales: List[float] = []

    def _done(self, f):
        return f in self.pose if "camera" in self.tasks else f in self.depth

    def _amp(self):
        use = self.device.type == "cuda" and self.amp_dtype is not None
        return torch.autocast("cuda", dtype=self.amp_dtype) if use else contextlib.nullcontext()

    def _head_ctx(self):
        use = self.device.type == "cuda" and self.amp_dtype is not None
        return torch.autocast("cuda", enabled=False) if (self.dtype == torch.float32 and use) else contextlib.nullcontext()

    def _heads(self, outs, imgs, psi):
        layers = head_inputs(self.model, outs, self.dtype)
        d = dc = None
        if "depth" in self.tasks:
            with self._head_ctx():
                d, dc = self.model.depth_head(layers, images=imgs.to(self.dtype), patch_start_idx=psi)
        last = [x for x in layers if x is not None][-1]
        return last[:, :, :1].clone(), d, dc

    def _emit(self, fresh, cam, d, dc, prev_keys, eligible=None):
        s = self.scales[-1] if self.scales else 1.0
        idx = {f: j for j, f in enumerate(fresh)}
        got = False
        if d is not None:
            ratios = [self.med[k] / float(d[:, idx[k]].float().median()) for k in prev_keys if k in idx and k in self.med]
            if ratios:
                s, got = float(min(max(float(np.median(ratios)), 1 / 3), 3.0)), True
        pe = None
        if "camera" in self.tasks:
            with self._head_ctx():
                pe = self.model.camera_head([cam])[-1].float()
            if not got:  # fall back to the ratio of translation norms of shared keys
                num = den = 0.0
                for k in prev_keys:
                    if k in idx and k in self.pose:
                        num += float(self.pose[k][:, :3].norm(dim=-1).mean())
                        den += float(pe[:, idx[k], :3].norm(dim=-1).mean())
                if den >= 0.3 and num > 0:
                    s = float(min(max(num / den, 1 / 3), 3.0))
        self.scales.append(s)
        new = []
        for j, f in enumerate(fresh):
            if eligible is not None and f not in eligible:
                continue
            if pe is not None and f not in self.pose:
                q = pe[:, j].clone()
                q[:, :3] *= s
                self.pose[f] = q
                new.append(f)
            if d is not None and f not in self.depth:
                self.depth[f] = (d[:, j].float() * s).cpu()
                self.conf[f] = dc[:, j].cpu()
                self.med[f] = float(d[:, j].float().median()) * s
                if pe is None:
                    new.append(f)
        return new

    def _attach(self, groups, cache, psi, fresh, cams, ds, dcs):
        for fr in groups:
            for s0 in range(0, len(fr), self.chunk):
                sub = fr[s0:s0 + self.chunk]
                imgs = self.buf.stack(sub)
                outs = attach_pass(self.model, imgs, cache)
                c, d, dc = self._heads(outs, imgs, psi)
                del outs
                fresh += sub
                cams.append(c); ds.append(d); dcs.append(dc)

    def _interval(self, i):
        return list(range(self.candidates[i - 1] + 1, self.candidates[i])) if i >= 1 else []

    @torch.no_grad()
    def _pass(self, m, extra_intervals):
        keys = _thin(self.candidates, m, self.max_keys)
        kset, prev_keys = set(keys), self.candidates[:m]
        imgs_k = self.buf.stack(keys)
        cache, outs_k, psi, _ = pin_pass(self.model, imgs_k)
        cam_k, d_k, dc_k = self._heads(outs_k, imgs_k, psi)
        del outs_k
        fresh, cams, ds, dcs = list(keys), [cam_k], [d_k], [dc_k]
        due = self.candidates[:max(m - self.lag, 0) + 1]
        eligible = set(due)
        missing = [f for f in due if f not in kset and not self._done(f)]
        groups = [self._interval(i) for i in extra_intervals] + ([missing] if missing else [])
        for fr in groups:
            eligible |= set(fr)
        self._attach(groups, cache, psi, fresh, cams, ds, dcs)
        return self._emit(fresh, torch.cat(cams, 1), torch.cat(ds, 1) if ds[0] is not None else None,
                          torch.cat(dcs, 1) if dcs[0] is not None else None, prev_keys, eligible)

    def push(self, image) -> List[int]:
        """Add the next frame ([3, H, W] in [0, 1], same size for all frames). Returns the indices
        of the frames emitted by this call."""
        i = self.n
        self.buf.add(i, image)
        self.n += 1
        if i % self.kappa != 0:
            return []
        self.candidates.append(i)
        m = len(self.candidates) - 1
        with self._amp():
            new = self._pass(m, [m - self.lag] if m - self.lag >= 1 else [])
        self.buf.release(set(self.pose) | set(self.depth), self.candidates)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return new

    @torch.no_grad()
    def finish(self) -> List[int]:
        """Emit every frame that is still pending after the last frame has arrived."""
        keys = _thin(self.candidates, len(self.candidates) - 1, self.max_keys)
        rest = [f for f in range(self.n) if not self._done(f) and f not in set(keys)]
        if not rest and all(self._done(k) for k in keys):
            return []
        with self._amp():
            imgs_k = self.buf.stack(keys)
            cache, outs_k, psi, _ = pin_pass(self.model, imgs_k)
            cam_k, d_k, dc_k = self._heads(outs_k, imgs_k, psi)
            del outs_k
            fresh, cams, ds, dcs = list(keys), [cam_k], [d_k], [dc_k]
            self._attach([rest] if rest else [], cache, psi, fresh, cams, ds, dcs)
            return self._emit(fresh, torch.cat(cams, 1), torch.cat(ds, 1) if ds[0] is not None else None,
                              torch.cat(dcs, 1) if dcs[0] is not None else None, self.candidates[:-1])

    def results(self) -> Dict:
        """All emitted outputs stacked in frame order (call after finish())."""
        out = {}
        if "camera" in self.tasks:
            out["pose_enc"] = torch.stack([self.pose[f] for f in range(self.n)], 1).to(self.dtype)
        if "depth" in self.tasks:
            out["depth"] = torch.stack([self.depth[f] for f in range(self.n)], 1)
            out["depth_conf"] = torch.stack([self.conf[f] for f in range(self.n)], 1)
        out["scales"] = list(self.scales)
        return out


class Pi3OnlineStream:
    """Online Stream on a pi3 backbone. pi3 has no reference frame, so each pass is aligned to
    the emitted frames with a similarity transform fitted on the camera centers and axes of
    the shared keys (scale from their median depth)."""

    def __init__(self, model, key_interval=16, max_keys=32, chunk=64, amp_dtype=torch.bfloat16):
        self.model, self.kappa, self.max_keys, self.chunk = model, key_interval, max_keys, chunk
        self.device = next(model.parameters()).device
        self.amp_dtype = amp_dtype
        self.buf = _Buffer(self.device)
        self.n = 0
        self.candidates: List[int] = []
        self.out = {k: {} for k in ("points", "local_points", "conf", "camera_poses")}
        self.med: Dict[int, float] = {}
        self.scales: List[float] = []
        self.prev = None
        self.last = None  # (keys, head outputs, cache) of the latest pass, reused for the tail

    def _amp(self):
        use = self.device.type == "cuda" and self.amp_dtype is not None
        return torch.autocast("cuda", dtype=self.amp_dtype) if use else contextlib.nullcontext()

    @staticmethod
    def _axes(c2w, k=0.3):
        c = c2w[:, :3, 3][:, None]
        return np.concatenate([c, c + k * c2w[:, :3, :3].transpose(0, 2, 1)], 1).reshape(-1, 3)

    def _emit(self, res, fresh, prev_keys):
        cp = res["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        idx = {f: j for j, f in enumerate(fresh)}
        shared = [k for k in prev_keys if k in idx and k in self.out["camera_poses"]]
        sc = self.scales[-1] if self.scales else 1.0
        R, t = np.eye(3), np.zeros(3)
        got = False
        if shared:
            z_now = res["local_points"][0, ..., 2].float()
            ratios = [self.med[k] / float(z_now[idx[k]].median()) for k in shared if k in self.med]
            if ratios:
                sc, got = float(min(max(float(np.median(ratios)), 1 / 3), 3.0)), True
            Cs = np.stack([cp[idx[k]] for k in shared])
            Cd = np.stack([self.out["camera_poses"][k][0].double().numpy() for k in shared])
            if not got:  # fall back to the spread of the shared camera centers
                ss = np.linalg.norm(Cs[:, :3, 3] - Cs[:, :3, 3].mean(0), axis=1).sum()
                sd = np.linalg.norm(Cd[:, :3, 3] - Cd[:, :3, 3].mean(0), axis=1).sum()
                if ss >= 0.3 and sd > 0:
                    sc = float(min(max(sd / ss, 1 / 3), 3.0))
            Cs2 = Cs.copy()
            Cs2[:, :3, 3] *= sc
            _, R, t = umeyama_alignment(self._axes(Cs2).T, self._axes(Cd).T, False)
        self.scales.append(float(sc))
        Rt = torch.tensor(R, dtype=torch.float32)
        tt = torch.tensor(t, dtype=torch.float32)
        new = []
        for j, f in enumerate(fresh):
            if f in self.out["camera_poses"]:
                continue
            c = res["camera_poses"][:, j].float().cpu()
            c2 = c.clone()
            c2[:, :3, :3] = Rt @ c[:, :3, :3]
            c2[:, :3, 3] = (sc * (Rt @ c[:, :3, 3].T).T) + tt
            self.out["camera_poses"][f] = c2
            self.out["local_points"][f] = res["local_points"][:, j].float().cpu() * sc
            self.out["conf"][f] = res["conf"][:, j].cpu()
            self.out["points"][f] = (res["points"][:, j].float().cpu() @ (sc * Rt).T) + tt
            self.med[f] = float(res["local_points"][0, j, ..., 2].float().median()) * sc
            new.append(f)
        return new

    def _run_pass(self, keys, attach_frames):
        res_k, cache = pi3_engine.run(self.model, self.buf.stack(keys))
        fresh, parts = list(keys), [res_k]
        for s0 in range(0, len(attach_frames), self.chunk):
            sub = attach_frames[s0:s0 + self.chunk]
            r, _ = pi3_engine.run(self.model, self.buf.stack(sub), cache=cache)
            parts.append(r)
            fresh += sub
        return fresh, parts, res_k, cache

    @torch.no_grad()
    def push(self, image) -> List[int]:
        i = self.n
        self.buf.add(i, image.float())
        self.n += 1
        if i % self.kappa != 0:
            return []
        self.candidates.append(i)
        m = len(self.candidates) - 1
        keys = _thin(self.candidates, m, self.max_keys)
        with self._amp():
            fresh, parts, res_k, cache = self._run_pass(keys, list(range(self.prev + 1, i)) if self.prev is not None else [])
            new = self._emit({k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}, fresh, self.candidates[:m])
        self.prev = i
        self.last = (keys, res_k, cache)
        self.buf.release(set(self.out["camera_poses"]), self.candidates)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return new

    @torch.no_grad()
    def finish(self) -> List[int]:
        keys, res_k, cache = self.last
        tail = [f for f in range(self.n) if f not in self.out["camera_poses"] and f not in set(keys)]
        if not tail:
            return []
        with self._amp():
            fresh, parts = list(keys), [res_k]
            for s0 in range(0, len(tail), self.chunk):
                sub = tail[s0:s0 + self.chunk]
                r, _ = pi3_engine.run(self.model, self.buf.stack(sub), cache=cache)
                parts.append(r)
                fresh += sub
            return self._emit({k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}, fresh, self.candidates[:-1])

    def results(self) -> Dict:
        """Emitted pi3 outputs stacked in frame order (same keys as pi3_engine.reconstruct_raw)."""
        ret = {k: torch.stack([v[f] for f in range(self.n)], 1) for k, v in self.out.items()}
        ret["scales"] = list(self.scales)
        return ret


def online_reconstruct_vggt(model, images, key_interval=16, lag=2, max_keys=32, chunk=64, tasks=("camera", "depth")):
    """Run the VGGT Online Stream over a whole clip [1, S, 3, H, W], frame by frame."""
    stream = VGGTOnlineStream(model, key_interval, lag, max_keys, chunk, tasks)
    for i in range(images.shape[1]):
        stream.push(images[0, i])
    stream.finish()
    return stream.results()


def online_reconstruct_pi3(model, images, key_interval=16, max_keys=32, chunk=64):
    """Run the pi3 Online Stream over a whole clip [1, S, 3, H, W], frame by frame."""
    stream = Pi3OnlineStream(model, key_interval, max_keys, chunk)
    for i in range(images.shape[1]):
        stream.push(images[0, i])
    stream.finish()
    return stream.results()
