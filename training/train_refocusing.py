r"""Train Alignment Refocusing: fine-tune global blocks 10 to 15 of VGGT, all other weights frozen.

Each step takes a whole scan or a random contiguous half, pins 24 evenly spaced frames, and
adds 24 random frames that read only the pinned keys. Loss: the VGGT camera loss on all
frames (weight 5) plus the depth loss on 6 random frames. Our weights: 2,000 AdamW steps,
learning rate 1e-5, 195 ScanNet scans, 3.6 hours on one A100 (80 GB).

    python training/train_refocusing.py --data_dir /data/scannet/process_scannet_train \
        --scenes training/scannet_train_195.txt --weights /path/to/VGGT-1B/model.pt --out runs/refocusing

Data comes from eval/prepare_scannet.py --train. Writes alignment_refocusing_step<N>.safetensors
(load with ecovggt.load_refocusing) and log.jsonl.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
import types
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecovggt.models import _add_path  # noqa: E402

_add_path("ECOVGGT_VGGT_ROOT", "FastVGGT")
_official = os.environ.get("ECOVGGT_VGGT_OFFICIAL_ROOT") or str(ROOT / "third_party" / "vggt")
sys.path.append(os.path.join(_official, "training"))  # VGGT training losses (loss.py)
if importlib.util.find_spec("iopath") is None:  # iopath is only used for file I/O in train_utils, so a stub is enough
    _m, _c, _f = types.ModuleType("iopath"), types.ModuleType("iopath.common"), types.ModuleType("iopath.common.file_io")
    _f.g_pathmgr = None
    sys.modules.update({"iopath": _m, "iopath.common": _c, "iopath.common.file_io": _f})
from loss import compute_camera_loss, compute_depth_loss  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402

from ecovggt.vggt_engine import CACHED_BLOCKS, _embed, cached_attention_block  # noqa: E402

BAND = range(10, 16)


# ----------------------------------------------------------------------------- data
class ScanNetScans:
    """Scans exported by eval/prepare_scannet.py --train: color/<i>.jpg, depth/<i>.png (mm),
    pose/<i>.txt (camera to world), intrinsic/intrinsic_{color,depth}.txt."""

    def __init__(self, data_dir, names=None, min_frames=64):
        self.scans = []
        for d in sorted(glob.glob(os.path.join(data_dir, "*"))):
            if names is not None and os.path.basename(d) not in names:
                continue
            if not (os.path.isdir(f"{d}/pose") and os.path.exists(f"{d}/intrinsic/intrinsic_depth.txt")):
                continue
            ids = sorted(int(f[:-4]) for f in os.listdir(f"{d}/pose") if f.endswith(".txt"))
            poses, valid = [], []
            for i in ids:
                if not (os.path.exists(f"{d}/depth/{i}.png") and os.path.exists(f"{d}/color/{i}.jpg")):
                    continue
                m = np.loadtxt(f"{d}/pose/{i}.txt", dtype=np.float64)
                if m.shape == (4, 4) and np.all(np.isfinite(m)):
                    poses.append(m)
                    valid.append(i)
            if len(valid) < min_frames:
                continue
            Kc, Kd = np.loadtxt(f"{d}/intrinsic/intrinsic_color.txt"), np.loadtxt(f"{d}/intrinsic/intrinsic_depth.txt")
            self.scans.append(dict(dir=d, name=os.path.basename(d), valid=np.array(valid), c2w=np.stack(poses), Kc=Kc[:3, :3], Kd=Kd[:3, :3]))
        assert self.scans, f"no usable scans in {data_dir}"
        print(f"[data] {len(self.scans)} scans", flush=True)

    def sample(self, n_pinned, n_query, rng, whole_prob=0.6):
        """Pinned frames spread over the whole scan or a random contiguous half, plus random other
        frames from that range. Returns the scan, the sorted frame ids, and the positions of the
        pinned frames (the first one is frame 0 of the sample)."""
        s = self.scans[rng.integers(len(self.scans))]
        v, n = s["valid"], len(s["valid"])
        if rng.random() < whole_prob:
            lo, hi = 0, n
        else:
            L = max(n // 2, n_pinned + n_query + 8)
            lo = rng.integers(0, n - L + 1)
            hi = lo + L
        pinned = np.round(np.linspace(lo, hi - 1, n_pinned)).astype(int)
        pool = np.array([i for i in range(lo, hi) if i not in set(pinned.tolist())])
        query = rng.choice(pool, size=min(n_query, len(pool)), replace=False)
        sel = np.array(sorted(set(pinned.tolist()) | set(query.tolist())))
        return s, v[sel], [int(np.where(sel == k)[0][0]) for k in pinned]

    @staticmethod
    def load(s, frame_ids):
        """images [S, 3, H, W] in [0, 1], depth [S, H, W] (m, 0 = invalid), K [S, 3, 3], c2w [S, 4, 4]."""
        from PIL import Image
        paths = [f"{s['dir']}/color/{i}.jpg" for i in frame_ids]
        images = load_and_preprocess_images(paths)
        H, W = images.shape[-2:]
        with Image.open(paths[0]) as im0:
            cw, ch = im0.size
        deps = []
        for i in frame_ids:
            x = np.array(Image.open(f"{s['dir']}/depth/{i}.png")).astype(np.float32) / 1000.0
            x[(x > 10) | (x < 1e-3)] = 0
            deps.append(x)
        Kc = torch.tensor(s["Kc"], dtype=torch.float32).clone()
        Kc[0] *= W / cw
        Kc[1] *= H / ch
        dep = depth_to_color(torch.from_numpy(np.stack(deps)), torch.tensor(s["Kd"], dtype=torch.float32), Kc, H, W)
        idx = [int(np.where(s["valid"] == i)[0][0]) for i in frame_ids]
        return images, dep, Kc.expand(len(frame_ids), 3, 3).clone(), torch.from_numpy(s["c2w"][idx]).float()


def depth_to_color(dep, Kd, Kc, H, W):
    """Reproject depth maps from the depth camera (intrinsics Kd) into the color camera
    (intrinsics Kc at H x W), keeping the nearest depth per pixel. ScanNet aligns both cameras."""
    S, dh, dw = dep.shape
    out = torch.zeros(S, H, W)
    v, u = torch.meshgrid(torch.arange(dh, dtype=torch.float32), torch.arange(dw, dtype=torch.float32), indexing="ij")
    for si in range(S):
        z = dep[si]
        m = z > 0
        X, Y = (u - Kd[0, 2]) / Kd[0, 0] * z, (v - Kd[1, 2]) / Kd[1, 1] * z
        uc = (Kc[0, 0] * X / z.clamp(min=1e-6) + Kc[0, 2]).round().long()
        vc = (Kc[1, 1] * Y / z.clamp(min=1e-6) + Kc[1, 2]).round().long()
        ok = m & (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
        zz, uu, vv = z[ok], uc[ok], vc[ok]
        order = torch.argsort(zz, descending=True)  # far to near, so near points overwrite
        out[si].view(-1)[vv[order] * W + uu[order]] = zz[order]
    return out


def make_batch(images, dep, K, c2w, device):
    """Normalization of VGGT training: the first camera is the origin, and the scale is the
    mean distance of the valid points to it. Returns the batch that the losses expect."""
    S, H, W = dep.shape
    w2c = torch.linalg.inv(c2w)
    E = (w2c @ c2w[0][None])[:, :3]
    mask = dep > 0
    ys, xs = torch.meshgrid(torch.arange(0, H, 4), torch.arange(0, W, 4), indexing="ij")
    d = dep[:, ys, xs]
    m = d > 0
    pts = torch.stack([(xs[None] - K[:, 0:1, 2:3]) / K[:, 0:1, 0:1] * d, (ys[None] - K[:, 1:2, 2:3]) / K[:, 1:2, 1:2] * d, d], -1)
    cam2first = w2c[0:1] @ c2w
    pts0 = torch.einsum("sij,shwj->shwi", cam2first[:, :3, :3], pts) + cam2first[:, :3, 3][:, None, None]
    scale = float(((pts0.norm(dim=-1) * m).sum() / (m.sum() + 1e-3)).clamp(1e-6, 1e6))
    E = E.clone()
    E[:, :3, 3] /= scale
    return dict(images=images[None].to(device), extrinsics=E[None].to(device), intrinsics=K[None].to(device),
                depths=(dep / scale)[None].to(device), point_masks=mask[None].to(device)), scale


# ----------------------------------------------------------------------------- model
def forward_pinned(model, images, pinned):
    """Aggregator pass in which every frame takes keys and values from the pinned frames only.
    Blocks before the band run without gradients, the band keeps its activations, and later
    frozen blocks are recomputed in the backward pass. Returns the head inputs and the patch
    start index."""
    agg = model.aggregator
    B, S = images.shape[:2]
    with torch.no_grad():
        tokens, pos, psi, P, C = _embed(model, images, has_reference=True)
    T = psi + P
    kv = torch.as_tensor(pinned, device=images.device)

    def run_block(i, tokens):
        tokens = agg.frame_blocks[i](tokens.view(B * S, T, C), pos=pos.view(B * S, T, 2))
        fo = tokens
        t, p = tokens.view(B, S, T, C), pos.view(B, S, T, 2)
        out = cached_attention_block(agg.global_blocks[i], t.reshape(B, S * T, C), t[:, kv].reshape(B, -1, C),
                                     p.reshape(B, S * T, 2), p[:, kv].reshape(B, -1, 2))
        return out.reshape(B * S, T, C), fo, out

    outs = {}
    for i in range(agg.aa_block_num):
        if i < min(BAND):
            with torch.no_grad():
                tokens, fo, go = run_block(i, tokens)
        elif i in BAND:
            tokens, fo, go = run_block(i, tokens)
        else:
            tokens, fo, go = checkpoint(run_block, i, tokens, use_reentrant=False)
        if i in CACHED_BLOCKS:
            outs[i] = torch.cat([fo.view(B, S, T, C), go.view(B, S, T, C)], -1)
    return [outs[i] for i in CACHED_BLOCKS], psi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--scenes", default=None, help="optional list of scan names (default: every scan in data_dir)")
    ap.add_argument("--weights", required=True, help="VGGT-1B model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n_pinned", type=int, default=24)
    ap.add_argument("--n_query", type=int, default=24)
    ap.add_argument("--depth_frames", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--w_cam", type=float, default=5.0)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import save_file
    import torchvision.transforms as TT

    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    names = set(l.strip() for l in open(a.scenes) if l.strip()) if a.scenes else None
    data = ScanNetScans(a.data_dir, names)

    model = VGGT(enable_point=False, **({"merging": None} if "merging" in VGGT.__init__.__code__.co_varnames else {"enable_track": False}))
    model.load_state_dict(torch.load(a.weights, map_location="cpu"), strict=False)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = {}
    for i in BAND:
        for n, p in model.aggregator.global_blocks[i].named_parameters():
            p.requires_grad_(True)
            trainable[f"aggregator.global_blocks.{i}.{n}"] = p
    print(f"[model] training global blocks {BAND.start} to {BAND.stop - 1}: {sum(p.numel() for p in trainable.values()) / 1e6:.1f}M parameters", flush=True)

    opt = torch.optim.AdamW(list(trainable.values()), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999))
    lr_at = lambda st: a.lr * min(1.0, (st + 1) / max(1, a.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, st / max(1, a.steps)))))
    jitter = TT.ColorJitter(0.3, 0.3, 0.2, 0.03)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t_start = time.time()
    for step in range(a.steps):
        t0 = time.time()
        scan, fids, pinned = data.sample(a.n_pinned, a.n_query, rng)
        images, dep, K, c2w = ScanNetScans.load(scan, fids)
        if rng.random() < 0.5:
            images = torch.stack([jitter(im) for im in images])
        batch, scale = make_batch(images, dep, K, c2w, dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        dsel = sorted(rng.choice(len(fids), size=min(a.depth_frames, len(fids)), replace=False).tolist())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            outs, psi = forward_pinned(model, batch["images"], pinned)
            pose_enc_list = model.camera_head(outs)
            d, dc = model.depth_head([o[:, dsel] for o in outs], images=batch["images"][:, dsel], patch_start_idx=psi)
        cam = compute_camera_loss({"pose_enc_list": [p.float() for p in pose_enc_list]}, batch, loss_type="l1", gamma=0.6,
                                  weight_trans=1.0, weight_rot=1.0, weight_focal=0.5)
        sub = {k: v[:, dsel] for k, v in batch.items()}
        dl = compute_depth_loss({"depth": d.float(), "depth_conf": dc.float()}, sub, gamma=1.0, alpha=0.2, gradient_loss_fn="grad", valid_range=0.98)
        depth_loss = dl["loss_conf_depth"] + dl["loss_reg_depth"] + dl["loss_grad_depth"]
        loss = a.w_cam * cam["loss_camera"] + depth_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), a.clip)
        opt.step()
        rec = dict(step=step, scan=scan["name"], frames=len(fids), loss=float(loss), camera=float(cam["loss_camera"]), depth=float(depth_loss),
                   grad_norm=float(gnorm), lr=lr_at(step), scale=scale, seconds=time.time() - t0,
                   memory_gb=torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 20 == 0:
            print(f"[{step:5d}] loss {rec['loss']:.4f} camera {rec['camera']:.4f} depth {rec['depth']:.4f} lr {rec['lr']:.1e} "
                  f"{scan['name']} {rec['seconds']:.1f}s {rec['memory_gb']:.1f}GB", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save_file({n: p.detach().to(torch.float32).cpu().contiguous() for n, p in trainable.items()},
                      os.path.join(a.out, f"alignment_refocusing_step{step + 1}.safetensors"),
                      metadata={"base": "facebook/VGGT-1B", "blocks": "aggregator.global_blocks.10-15", "step": str(step + 1)})
    print(f"done: {a.steps} steps in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
