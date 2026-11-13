"""EcoVGGT on Depth Anything 3 (DA3-LARGE).

From `alt_start` on, the odd layers attend across views. The pin pass caches their inputs
(frame 0 first, as the reference view), and attach batches read keys and values from them.

Layer loop adapted from Depth Anything 3 (https://github.com/ByteDance-Seed/Depth-Anything-3),
Apache License 2.0, see licenses/LICENSE-Depth-Anything-3-Apache-2.0.txt. Modified to cache
the inputs of the cross-view layers and to attend to this cache in attach batches.
"""
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from .keys import batches, pinned_keys

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _net(model):
    return model.model if hasattr(model, "model") else model


def _vit(model):
    net = _net(model)
    return net.backbone.pretrained, list(net.backbone.out_layers)


def normalize_images(images):
    return (images - IMNET_MEAN.to(images)) / IMNET_STD.to(images)


def _attn_kv(attn, xq, xkv, pos_q, pos_kv):
    B, Nq, C = xq.shape
    Nk = xkv.shape[1]
    h = attn.num_heads
    d = C // h
    q = attn.qkv(xq).reshape(B, Nq, 3, h, d).permute(2, 0, 3, 1, 4)[0]
    kv = attn.qkv(xkv).reshape(B, Nk, 3, h, d).permute(2, 0, 3, 1, 4)
    k, v = kv[1], kv[2]
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None and pos_q is not None:
        q, k = attn.rope(q, pos_q), attn.rope(k, pos_kv)
    x = F.scaled_dot_product_attention(q, k, v)
    return attn.proj_drop(attn.proj(x.transpose(1, 2).reshape(B, Nq, C)))


def cached_attention_block(blk, x, x_kv, pos_q, pos_kv):
    x = x + blk.ls1(_attn_kv(blk.attn, blk.norm1(x), blk.norm1(x_kv), pos_q, pos_kv))
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


def _cam_tokens(vit, B, S, ref_first):
    src = vit.camera_token[:, 1:].expand(B, S - (1 if ref_first else 0), -1)
    return torch.cat([vit.camera_token[:, :1].expand(B, -1, -1), src], 1) if ref_first else src


def _pack(vit, o):
    C = vit.embed_dim
    cam = o[:, :, 0]
    pt = torch.cat([o[..., :C], vit.norm(o[..., C:])], -1) if o.shape[-1] == 2 * C else vit.norm(o)
    return pt[:, :, 1 + vit.num_register_tokens:], cam


def _loop(vit, out_layers, x, pos, pos_nodiff, B, S, ref_first, cache=None, kv_cache=None):
    feats, local_x = {}, x
    for i, blk in enumerate(vit.blocks):
        g_pos, l_pos = (None, None) if (i < vit.rope_start or vit.rope is None) else (pos_nodiff, pos)
        if vit.alt_start != -1 and i == vit.alt_start:
            x = x.clone()
            x[:, :, 0] = _cam_tokens(vit, B, S, ref_first).to(x.dtype)
        if vit.alt_start != -1 and i >= vit.alt_start and i % 2 == 1:
            if kv_cache is not None:
                xk, pk = kv_cache[i]
                xq = rearrange(x, "b s n c -> b (s n) c")
                pq = rearrange(g_pos, "b s n c -> b (s n) c") if g_pos is not None else None
                x = rearrange(cached_attention_block(blk, xq, xk, pq, pk), "b (s n) c -> b s n c", s=S)
            else:
                if cache is not None:
                    cache[i] = (rearrange(x, "b s n c -> b (s n) c"), rearrange(g_pos, "b s n c -> b (s n) c") if g_pos is not None else None)
                x = vit.process_attention(x, blk, "global", pos=g_pos)
        else:
            x = vit.process_attention(x, blk, "local", pos=l_pos)
            local_x = x
        if i in out_layers:
            feats[i] = _pack(vit, torch.cat([local_x, x], -1) if vit.cat_token else x)
    return [feats[i] for i in out_layers]


@torch.no_grad()
def pin_pass(model, images_k):
    vit, out_layers = _vit(model)
    B, S, _, H, W = images_k.shape
    x = vit.prepare_tokens_with_masks(images_k)
    pos, pos_nodiff = vit._prepare_rope(B, S, H, W, images_k.device)
    cache = {}
    feats = _loop(vit, out_layers, x, pos, pos_nodiff, B, S, True, cache=cache)
    return cache, feats


@torch.no_grad()
def attach_pass(model, images_c, cache):
    vit, out_layers = _vit(model)
    B, c, _, H, W = images_c.shape
    x = vit.prepare_tokens_with_masks(images_c)
    pos, pos_nodiff = vit._prepare_rope(B, c, H, W, images_c.device)
    return _loop(vit, out_layers, x, pos, pos_nodiff, B, c, False, kv_cache=cache)


@torch.no_grad()
def heads(model, feats, H, W):
    net = _net(model)
    with torch.autocast("cuda", enabled=False):
        f32 = [(a.float(), b.float()) for a, b in feats]
        out = net.head(f32, H, W, patch_start_idx=0)
        pe = net.cam_dec(f32[-1][1]) if net.cam_dec is not None else None
    return out, pe


@torch.no_grad()
def reconstruct_raw(model, images, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None, chunk: int = 64) -> Dict:
    """images [1, S, 3, H, W] in [0, 1]. Returns DA3 head outputs per frame and the DA3 pose
    encoding (camera to world, canonical frame of the pin pass)."""
    B, S, _, H, W = images.shape
    imn = normalize_images(images)
    kf = [0] + sorted(set(key_frames if key_frames is not None else pinned_keys(S, num_keys)) - {0})
    cache, fk = pin_pass(model, imn[:, kf])
    outk, pek = heads(model, fk, H, W)
    del fk
    outs, pe = {k: [None] * S for k in outk}, [None] * S
    for j, f in enumerate(kf):
        for k in outk:
            outs[k][f] = outk[k][:, j].cpu()
        pe[f] = pek[:, j].float().cpu()
    rest = [f for f in range(S) if f not in set(kf)]
    for sub in batches(rest, chunk):
        fc = attach_pass(model, imn[:, sub], cache)
        outc, pec = heads(model, fc, H, W)
        del fc
        for j, f in enumerate(sub):
            for k in outc:
                outs[k][f] = outc[k][:, j].cpu()
            pe[f] = pec[:, j].float().cpu()
    res = {k: torch.stack(v, 1) for k, v in outs.items()}
    res["pose_enc_c2w"] = torch.stack(pe, 1)
    res["pinned_frames"] = kf
    return res


@torch.no_grad()
def to_camera_outputs(res, H, W, device):
    """Camera matrices relative to frame 0, intrinsics, depth, confidence and world points."""
    import depth_anything_3.model.utils.transform as T
    S = res["pose_enc_c2w"].shape[1]
    with torch.autocast("cuda", enabled=False):
        c2w, K = T.pose_encoding_to_extri_intri(res["pose_enc_c2w"].to(device).float(), (H, W))
        c2w = c2w.double()
        if c2w.shape[-2] == 3:
            c2w = torch.cat([c2w, torch.tensor([0., 0., 0., 1.], dtype=c2w.dtype, device=device).view(1, 1, 1, 4).expand(1, S, 1, 4)], -2)
        c2w_rel = torch.linalg.inv(c2w[:, :1]) @ c2w
        w2c_rel = torch.linalg.inv(c2w_rel)
        depth = res["depth"].to(device).float()
        conf = res["depth_conf"].to(device).float()
        u = torch.arange(W, device=device, dtype=torch.float64).view(1, 1, 1, W)
        v = torch.arange(H, device=device, dtype=torch.float64).view(1, 1, H, 1)
        Kd = K.double()
        fx, fy = Kd[..., 0, 0].view(1, S, 1, 1), Kd[..., 1, 1].view(1, S, 1, 1)
        cx, cy = Kd[..., 0, 2].view(1, S, 1, 1), Kd[..., 1, 2].view(1, S, 1, 1)
        z = depth.double()
        pc = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
        R = c2w_rel[:, :, :3, :3].view(1, S, 1, 1, 3, 3)
        t = c2w_rel[:, :, :3, 3].view(1, S, 1, 1, 3)
        pts = (R @ pc[..., None]).squeeze(-1) + t
    return dict(c2w=c2w_rel[0].float(), w2c=w2c_rel[0, :, :3].float(), intrinsics=K[0].float(), depth=depth[0], conf=conf[0], world_points=pts[0].float())


@torch.no_grad()
def reconstruct(model, images, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None, chunk: int = 64,
                amp_dtype=torch.bfloat16) -> Dict:
    """Run EcoVGGT on DA3. images: [1, S, 3, H, W] in [0, 1]."""
    H, W = images.shape[-2:]
    amp = torch.autocast("cuda", dtype=amp_dtype) if (images.is_cuda and amp_dtype is not None) else torch.autocast("cpu", enabled=False)
    with amp:
        raw = reconstruct_raw(model, images.float(), num_keys=num_keys, key_frames=key_frames, chunk=chunk)
    out = to_camera_outputs(raw, H, W, images.device)
    out["pinned_frames"] = raw["pinned_frames"]
    return out
