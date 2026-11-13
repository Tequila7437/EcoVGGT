"""EcoVGGT on pi3: Pinned Key Set inference with a Scene Cache.

pi3 has no reference frame and its heads decode every frame on its own, so the pin pass
and every attach batch give their outputs independently. Global layers of attach batches
read the cached tokens of the pinned frames.

The decoder loop and the head calls are adapted from pi3 (https://github.com/yyfz/Pi3),
Copyright (c) 2025, the authors, under the BSD 3-Clause License; see
licenses/LICENSE-Pi3-BSD-3-Clause.txt. The changes restrict the keys and values of the
global layers to cached pinned frames.
"""
from typing import Dict, Optional, Sequence

import torch

from .keys import batches, pinned_keys


def cached_attention(attn, x_q, x_kv, pos_q, pos_kv):
    """pi3 AttentionRope with queries from x_q and keys and values from x_kv."""
    B, Nq, C = x_q.shape
    Nk = x_kv.shape[1]
    H = attn.num_heads
    D = C // H
    qkv_q = attn.qkv(x_q).reshape(B, Nq, 3, H, D).permute(2, 0, 3, 1, 4)
    qkv_k = attn.qkv(x_kv).reshape(B, Nk, 3, H, D).permute(2, 0, 3, 1, 4)
    q = attn.q_norm(qkv_q[0]).to(qkv_q[2].dtype)
    k = attn.k_norm(qkv_k[1]).to(qkv_k[2].dtype)
    v = qkv_k[2]
    if attn.rope is not None:
        q, k = attn.rope(q, pos_q), attn.rope(k, pos_kv)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C)
    return attn.proj_drop(attn.proj(o))


def cached_attention_block(block, x_q, x_kv, pos_q, pos_kv):
    x = x_q + block.drop_path1(block.ls1(cached_attention(block.attn, block.norm1(x_q), block.norm1(x_kv), pos_q, pos_kv)))
    return x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))


def _embed(model, imgs):
    B, N, _, H, W = imgs.shape
    x = (imgs - model.image_mean) / model.image_std
    hidden = model.encoder(x.reshape(B * N, 3, H, W), is_training=True)
    if isinstance(hidden, dict):
        hidden = hidden["x_norm_patchtokens"]
    P = hidden.shape[1]
    psi = model.patch_start_idx
    reg = model.register_token.repeat(B, N, 1, 1).reshape(B * N, *model.register_token.shape[-2:])
    hidden = torch.cat([reg, hidden], 1)
    pos = model.position_getter(B * N, H // model.patch_size, W // model.patch_size, hidden.device) + 1
    pos = torch.cat([torch.zeros(B * N, psi, 2, device=hidden.device, dtype=pos.dtype), pos], 1)
    return hidden, pos, psi, P


def _heads(model, hidden, pos, psi, B, N, H, W):
    from pi3.utils.geometry import homogenize_points
    point_hidden = model.point_decoder(hidden, xpos=pos)
    conf_hidden = model.conf_decoder(hidden, xpos=pos)
    camera_hidden = model.camera_decoder(hidden, xpos=pos)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        r = model.point_head([point_hidden.float()[:, psi:]], (H, W)).reshape(B, N, H, W, -1)
        xy, z = r.split([2, 1], dim=-1)
        z = torch.exp(z)
        local_points = torch.cat([xy * z, z], -1)
        conf = model.conf_head([conf_hidden.float()[:, psi:]], (H, W)).reshape(B, N, H, W, -1)
        camera_poses = model.camera_head(camera_hidden.float()[:, psi:], H // 14, W // 14).reshape(B, N, 4, 4)
        points = torch.einsum("bnij, bnhwj -> bnhwi", camera_poses, homogenize_points(local_points))[..., :3]
    return dict(points=points, local_points=local_points, conf=conf, camera_poses=camera_poses)


@torch.no_grad()
def run(model, imgs, cache=None):
    """Pin pass (cache=None: full attention among imgs, returns a new Scene Cache) or attach
    pass (global layers read `cache`). Returns head outputs and the cache."""
    B, N, _, H, W = imgs.shape
    hidden, pos, psi, P = _embed(model, imgs)
    T = psi + P
    finals, new_cache = [], {}
    L = len(model.decoder)
    for i in range(L):
        blk = model.decoder[i]
        if i % 2 == 0:
            hidden = blk(hidden.reshape(B * N, T, -1), xpos=pos.reshape(B * N, T, -1))
        else:
            hq, pq = hidden.reshape(B, N * T, -1), pos.reshape(B, N * T, -1)
            if cache is None:
                new_cache[i] = (hq, pq)
                hidden = cached_attention_block(blk, hq, hq, pq, pq)
            else:
                hk, pk = cache[i]
                hidden = cached_attention_block(blk, hq, hk, pq, pk)
        hidden = hidden.reshape(B * N, T, -1)
        if i + 1 in (L - 1, L):
            finals.append(hidden)
    out = _heads(model, torch.cat(finals, -1), pos, psi, B, N, H, W)
    return out, (new_cache if cache is None else cache)


@torch.no_grad()
def reconstruct_raw(model, imgs, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None, chunk: int = 64) -> Dict:
    """imgs [1, S, 3, H, W] in [0, 1]. Returns pi3 outputs per frame (points, local_points,
    conf, camera_poses) in the canonical frame of the pin pass, plus the pinned frames."""
    B, S = imgs.shape[:2]
    keys = sorted(set(int(k) for k in (key_frames if key_frames is not None else pinned_keys(S, num_keys))))
    rest = [i for i in range(S) if i not in set(keys)]
    out = {k: [None] * S for k in ("points", "local_points", "conf", "camera_poses")}

    def put(res, frames):
        for k in out:
            for j, f in enumerate(frames):
                out[k][f] = res[k][:, j].cpu()

    res, cache = run(model, imgs[:, keys])
    put(res, keys)
    del res
    for fr in batches(rest, chunk):
        res, _ = run(model, imgs[:, fr], cache=cache)
        put(res, fr)
        del res
    ret = {k: torch.stack(v, 1) for k, v in out.items()}
    ret["pinned_frames"] = keys
    return ret


def fit_intrinsics(local_points, conf, H, W, q=0.1):
    """Pinhole intrinsics [S, 3, 3] from camera-frame points [S, H, W, 3]: principal point at the
    image center, focal lengths by least squares over pixels with sigmoid confidence above q."""
    S = local_points.shape[0]
    dev = local_points.device
    u = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W).expand(S, H, W)
    v = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1).expand(S, H, W)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    x, y, z = local_points.unbind(-1)
    m = (conf > q) & (z > 1e-3) & torch.isfinite(z)
    ax = torch.where(m, x / z.clamp_min(1e-6), 0.)
    ay = torch.where(m, y / z.clamp_min(1e-6), 0.)
    bx, by = torch.where(m, u - cx, 0.), torch.where(m, v - cy, 0.)
    fx = (ax * bx).flatten(1).sum(1) / (ax * ax).flatten(1).sum(1).clamp_min(1e-6)
    fy = (ay * by).flatten(1).sum(1) / (ay * ay).flatten(1).sum(1).clamp_min(1e-6)
    K = torch.zeros(S, 3, 3, device=dev)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, cx, cy, 1
    return K


@torch.no_grad()
def to_camera_outputs(out, H, W, device=None):
    """Express pi3 outputs in the camera frame of frame 0 and add per-frame intrinsics.

    Returns c2w [S, 4, 4], w2c [S, 3, 4], intrinsics [S, 3, 3], depth [S, H, W],
    conf [S, H, W] (sigmoid), world_points [S, H, W, 3]."""
    device = device or out["camera_poses"].device
    with torch.autocast("cuda", enabled=False):
        c2w = out["camera_poses"][0].float().to(device)
        c2w = torch.linalg.inv(c2w[0])[None] @ c2w
        lp = out["local_points"][0].float().to(device)
        conf = torch.sigmoid(out["conf"][0, ..., 0].float().to(device))
        K = fit_intrinsics(lp, conf, H, W)
        w2c = torch.linalg.inv(c2w)[:, :3]
        pts = out["points"][0].float().to(device)
        T0 = torch.linalg.inv(out["camera_poses"][0, 0].float().to(device))
        pts = pts @ T0[:3, :3].T + T0[:3, 3]
    return dict(c2w=c2w, w2c=w2c, intrinsics=K, depth=lp[..., 2], conf=conf, world_points=pts)


@torch.no_grad()
def reconstruct(model, imgs, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None, chunk: int = 64,
                amp_dtype=torch.bfloat16) -> Dict:
    """Reconstruct a video with EcoVGGT on a pi3 backbone (imgs [1, S, 3, H, W] in [0, 1])."""
    H, W = imgs.shape[-2:]
    amp = torch.autocast("cuda", dtype=amp_dtype) if (imgs.is_cuda and amp_dtype is not None) else torch.autocast("cpu", enabled=False)
    with amp:
        raw = reconstruct_raw(model, imgs.float(), num_keys=num_keys, key_frames=key_frames, chunk=chunk)
    res = to_camera_outputs(raw, H, W, device=imgs.device)
    res["pinned_frames"] = raw["pinned_frames"]
    res["raw"] = raw
    return res
