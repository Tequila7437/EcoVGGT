"""EcoVGGT on VGGT-Omega.

Global blocks attend over all tokens, and register blocks over the camera and register
tokens only. The pin pass caches the inputs of both kinds, and attach batches read keys and
values from the cache. The dense head runs per frame, and the camera head reads the last
camera and register tokens of all frames. Precision follows the official forward pass
(bf16 autocast for the aggregator, fp32 heads).

Layer loop adapted from VGGT-Omega (https://github.com/facebookresearch/vggt-omega), FAIR
Noncommercial Research License, see licenses/LICENSE-VGGT-Omega-FAIR-Noncommercial.txt.
Modified to cache the inputs of the cross-frame blocks for attach batches.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .keys import batches, pinned_keys


def _bias(qkv):
    b = qkv.bias
    if b is not None and hasattr(qkv, "bias_mask"):  # the official layer multiplies the whole bias by this mask
        b = b * qkv.bias_mask.to(b.dtype)
    return b


def _q(attn, x):
    C, h = attn.qkv.in_features, attn.num_heads
    b = _bias(attn.qkv)
    q = F.linear(x, attn.qkv.weight[:C], None if b is None else b[:C])
    q = q.reshape(x.shape[0], x.shape[1], h, C // h).transpose(1, 2)
    return attn.q_norm(q) if attn.use_qk_norm else q


def _kv(attn, x):
    C, h = attn.qkv.in_features, attn.num_heads
    b = _bias(attn.qkv)
    kv = F.linear(x, attn.qkv.weight[C:], None if b is None else b[C:]).reshape(x.shape[0], x.shape[1], 2, h, C // h)
    k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
    return (attn.k_norm(k) if attn.use_qk_norm else k), v


def cached_block(blk, x, x_kv):
    """Cross-frame block (no positional encoding) with queries from x and keys and values from x_kv."""
    a = blk.attn
    B, N, C = x.shape
    o = F.scaled_dot_product_attention(_q(a, blk.norm1(x)), *_kv(a, blk.norm1(x_kv))).transpose(1, 2).reshape(B, N, C)
    x = x + blk.ls1(a.proj_drop(a.proj(o)))
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


def _tokens(agg, images, has_reference):
    B, S, _, H, W = images.shape
    x = ((images - agg._resnet_mean) / agg._resnet_std).view(B * S, 3, H, W)
    if has_reference:
        cam = torch.cat([agg.camera_token[:, 0:1].expand(B, 1, -1, -1), agg.camera_token[:, 1:].expand(B, S - 1, -1, -1)], 1)
        reg = torch.cat([agg.register_token[:, 0:1].expand(B, 1, -1, -1), agg.register_token[:, 1:].expand(B, S - 1, -1, -1)], 1)
    else:
        cam, reg = agg.camera_token[:, 1:].expand(B, S, -1, -1), agg.register_token[:, 1:].expand(B, S, -1, -1)
    pt = agg.patch_embed(x)
    if isinstance(pt, dict):
        pt = pt["x_norm_patchtokens"]
    tok = torch.cat([cam.reshape(B * S, *cam.shape[2:]), reg.reshape(B * S, *reg.shape[2:]), pt], 1)
    with torch.no_grad():
        rs, rc = agg.rope_embed(H=H // agg.patch_size, W=W // agg.patch_size)
    rope = (rs.to(device=pt.device, dtype=torch.float32), rc.to(device=pt.device, dtype=torch.float32))
    return tok, rope, (B, S, tok.shape[1], tok.shape[2])


def _loop(agg, images, has_reference, cache=None, kv_cache=None):
    """Layer loop of the aggregator. With `cache`, records the inputs of the cross-frame blocks
    (pin pass); with `kv_cache`, cross-frame blocks read keys and values from it (attach)."""
    tok, rope, (B, S, T, C) = _tokens(agg, images, has_reference)
    P = agg.patch_token_start
    outs = []
    for b in range(agg.depth):
        tok = agg.frame_blocks[b](tok.view(B * S, T, C), rope)
        frame_tok = tok.view(B, S, T, C)
        blk, x = agg.inter_frame_blocks[b], tok.view(B, S, T, C)
        if agg.inter_frame_attention_types[b] == "global":
            xg = x.reshape(B, S * T, C)
            if kv_cache is not None:
                xg = cached_block(blk, xg, kv_cache[b])
            else:
                if cache is not None:
                    cache[b] = xg
                xg = blk(xg, None)
            x = xg.view(B, S, T, C)
        else:  # register block: only the camera and register tokens attend across frames
            cr, pt = x[:, :, :P].reshape(B, S * P, C), x[:, :, P:]
            if kv_cache is not None:
                cr = cached_block(blk, cr, kv_cache[b])
            else:
                if cache is not None:
                    cache[b] = cr
                cr = blk(cr, None)
            x = torch.cat([cr.view(B, S, P, C), pt], 2)
        tok = x
        outs.append(torch.cat([frame_tok, x], -1) if b in agg.cached_layer_indices else None)
    return outs


def _dense(model, outs, images):
    with torch.autocast(device_type="cuda", enabled=False):
        return model.dense_head(outs, images=images, patch_token_start=model.aggregator.patch_token_start)


def _camreg(outs, P):
    x = outs[-1][:, :, :P]
    return x.float() if x.dtype != torch.float32 else x.clone()


def _camera(model, camreg):
    with torch.autocast(device_type="cuda", enabled=False):
        return model.camera_head([camreg], patch_token_start=camreg.shape[2])


def _amp_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _to_gpu(x):
    return x if x.is_cuda else x.to("cuda", non_blocking=True)


@torch.no_grad()
def reconstruct_raw(model, images, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None, chunk: int = 64) -> Dict:
    """images: [1, S, 3, h, w] in [0, 1] from the VGGT-Omega preprocessing. They may stay on the
    CPU, since batches are moved to the GPU. Returns the VGGT-Omega pose encoding [1, S, 9],
    depth [1, S, h, w, 1], depth confidence [1, S, h, w], and the pinned frames."""
    agg = model.aggregator
    P, S = agg.patch_token_start, images.shape[1]
    keys = [0] + sorted(set(key_frames if key_frames is not None else pinned_keys(S, num_keys)) - {0})
    depth, conf, cr = [None] * S, [None] * S, [None] * S
    cache = {}
    xk = _to_gpu(images[:, keys])
    with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
        outs = _loop(agg, xk, True, cache=cache)
    d, c = _dense(model, outs, xk)
    crk = _camreg(outs, P)
    del outs, xk
    for j, f in enumerate(keys):
        depth[f], conf[f], cr[f] = d[:, j].cpu(), c[:, j].cpu(), crk[:, j]
    for sub in batches([f for f in range(S) if f not in set(keys)], chunk):
        xc = _to_gpu(images[:, sub])
        with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
            outs = _loop(agg, xc, False, kv_cache=cache)
        d, c = _dense(model, outs, xc)
        crc = _camreg(outs, P)
        del outs, xc
        for j, f in enumerate(sub):
            depth[f], conf[f], cr[f] = d[:, j].cpu(), c[:, j].cpu(), crc[:, j]
    del cache
    pose = _camera(model, torch.stack(cr, 1))
    d = torch.stack(depth, 1)
    return dict(pose_enc=pose.float(), depth=d if d.dim() == 5 else d[..., None], depth_conf=torch.stack(conf, 1), pinned_frames=keys)


@torch.no_grad()
def online_reconstruct(model, images, key_interval: int = 16, lag: int = 2, max_keys: int = 32, chunk: int = 64) -> Dict:
    """Online Stream over a clip [1, S, 3, h, w] in arrival order. Each pass uses only frames
    that have arrived, and scales are matched with the median depth of shared keys, as for
    VGGT. Returns the same entries as reconstruct_raw."""
    agg = model.aggregator
    P, S = agg.patch_token_start, images.shape[1]
    cands = list(range(0, S, key_interval))
    pose, depth, conf = [None] * S, [None] * S, [None] * S
    med: Dict[int, float] = {}
    scales: List[float] = []

    def keyset(m):
        c = cands[:m + 1]
        return c if not max_keys or len(c) <= max_keys else [c[i] for i in np.round(np.linspace(0, m, max_keys)).astype(int)]

    def run(frames, has_reference, cache=None, kv_cache=None):
        xb = _to_gpu(images[:, frames])
        with torch.autocast(device_type="cuda", dtype=_amp_dtype()):
            outs = _loop(agg, xb, has_reference, cache=cache, kv_cache=kv_cache)
        d, c = _dense(model, outs, xb)
        cr = _camreg(outs, P)
        del outs, xb
        return cr, (d if d.dim() == 5 else d[..., None]), c

    def emit(fresh, cr, d, c, prev_keys, eligible):
        idx = {f: j for j, f in enumerate(fresh)}
        s = scales[-1] if scales else 1.0
        r = [med[k] / float(d[:, idx[k]].float().median()) for k in prev_keys if k in idx and k in med]
        if r:
            s = float(min(max(float(np.median(r)), 1 / 3), 3.0))
        pe = _camera(model, cr).float()
        scales.append(s)
        for j, f in enumerate(fresh):
            if eligible is not None and f not in eligible:
                continue
            if pose[f] is None:
                q = pe[:, j].clone()
                q[:, :3] *= s
                pose[f] = q
            if depth[f] is None:
                depth[f] = (d[:, j].float() * s).cpu()
                conf[f] = c[:, j].float().cpu()
                med[f] = float(d[:, j].float().median()) * s

    def interval(i):
        return list(range(cands[i - 1] + 1, cands[i])) if i >= 1 else []

    def attach_and_emit(keys, groups, cache, cr_k, d_k, c_k, prev_keys, eligible):
        fresh, crs, ds, cs = list(keys), [cr_k], [d_k], [c_k]
        for fr in groups:
            if eligible is not None:
                eligible |= set(fr)
            for sub in batches(fr, chunk):
                cr_, d_, c_ = run(sub, False, kv_cache=cache)
                fresh += sub
                crs.append(cr_)
                ds.append(d_)
                cs.append(c_)
        emit(fresh, torch.cat(crs, 1), torch.cat(ds, 1), torch.cat(cs, 1), prev_keys, eligible)

    for m in range(len(cands)):
        keys, cache = keyset(m), {}
        cr_k, d_k, c_k = run(keys, True, cache=cache)
        due = cands[:max(m - lag, 0) + 1]
        missing = [f for f in due if f not in set(keys) and pose[f] is None]
        groups = ([interval(m - lag)] if m - lag >= 1 else []) + ([missing] if missing else [])
        attach_and_emit(keys, groups, cache, cr_k, d_k, c_k, cands[:m], set(due))
        del cache
        torch.cuda.empty_cache()
    keys = keyset(len(cands) - 1)
    rest = [f for f in range(S) if pose[f] is None and f not in set(keys)]
    if rest or any(pose[f] is None for f in keys):  # end of stream: output the rest
        cache = {}
        cr_k, d_k, c_k = run(keys, True, cache=cache)
        attach_and_emit(keys, [rest] if rest else [], cache, cr_k, d_k, c_k, cands[:-1], None)
        del cache
    return dict(pose_enc=torch.stack(pose, 1), depth=torch.stack(depth, 1), depth_conf=torch.stack(conf, 1), pinned_frames=cands, scales=scales)


@torch.no_grad()
def to_camera_outputs(raw, h, w):
    """World-to-camera matrices [S, 3, 4] relative to frame 0 (OpenCV), intrinsics [S, 3, 3] in
    pixels of the h x w input, depth [S, h, w], confidence [S, h, w], and world points."""
    from vggt_omega.utils.pose_enc import encoding_to_camera
    dev = raw["pose_enc"].device
    w2c, K = encoding_to_camera(raw["pose_enc"], (h, w))
    w2c, K = w2c.double().cpu(), K.double().cpu()
    S = w2c.shape[1]
    w2c4 = torch.cat([w2c, torch.tensor([0., 0., 0., 1.], dtype=torch.float64).view(1, 1, 1, 4).expand(1, S, 1, 4)], -2)
    c2w = torch.linalg.inv(w2c4)
    c2w_rel = torch.linalg.inv(c2w[:, :1]) @ c2w
    depth = raw["depth"][0, ..., 0].float().cpu()
    u = torch.arange(w, dtype=torch.float64).view(1, 1, w)
    v = torch.arange(h, dtype=torch.float64).view(1, h, 1)
    Kd = K[0]
    z = depth.double()
    pc = torch.stack([(u - Kd[:, 0, 2].view(S, 1, 1)) / Kd[:, 0, 0].view(S, 1, 1) * z,
                      (v - Kd[:, 1, 2].view(S, 1, 1)) / Kd[:, 1, 1].view(S, 1, 1) * z, z], -1)
    pts = (c2w_rel[0, :, :3, :3].view(S, 1, 1, 3, 3) @ pc[..., None]).squeeze(-1) + c2w_rel[0, :, :3, 3].view(S, 1, 1, 3)
    return dict(w2c=torch.linalg.inv(c2w_rel)[0, :, :3].float(), intrinsics=K[0].float(), depth=depth,
                conf=raw["depth_conf"][0].float().cpu(), world_points=pts.float(), device=dev)


@torch.no_grad()
def reconstruct(model, images, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None, chunk: int = 64) -> Dict:
    """Run EcoVGGT on VGGT-Omega. images: [1, S, 3, h, w] from
    load_images(..., backbone="vggt-omega")."""
    raw = reconstruct_raw(model, images, num_keys=num_keys, key_frames=key_frames, chunk=chunk)
    out = to_camera_outputs(raw, *images.shape[-2:])
    out["pinned_frames"] = raw["pinned_frames"]
    out["raw"] = raw
    return out
