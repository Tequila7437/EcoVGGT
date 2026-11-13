"""EcoVGGT on VGGT: Pinned Key Set inference with a Scene Cache.

A pin pass runs all 24 blocks on the pinned frames with full attention among them and
stores the input tokens of every global block (the Scene Cache). Every other frame then
runs in independent batches whose global blocks attend only to the cached pinned tokens.
This reproduces joint inference with the pinned frames as the only keys exactly, at a
memory cost that does not grow with the length of the video.
"""
import contextlib
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from .alignment import SCORE_BLOCK, NUM_QUERIES, attention_divergence
from .keys import batches, pinned_keys

CACHED_BLOCKS = (4, 11, 17, 23)  # blocks whose outputs the camera and depth heads read


def _special_tokens(tok, B, S, has_reference):
    """Camera and register tokens [B*S, X, C]. VGGT gives the first frame of a sequence
    its own (reference) tokens. An attach batch never contains frame 0, so all of its
    frames take the tokens of non-reference frames."""
    if has_reference:
        query = tok[:, 0:1].expand(B, 1, *tok.shape[2:])
        others = tok[:, 1:].expand(B, S - 1, *tok.shape[2:])
        return torch.cat([query, others], dim=1).view(B * S, *tok.shape[2:])
    return tok[:, 1:2].expand(B, S, *tok.shape[2:]).reshape(B * S, *tok.shape[2:])


def _embed(model, images, has_reference):
    agg = model.aggregator
    B, S, C_in, H, W = images.shape
    x = agg.patch_embed(((images - agg._resnet_mean) / agg._resnet_std).view(B * S, C_in, H, W))
    if isinstance(x, dict):
        x = x["x_norm_patchtokens"]
    _, P, C = x.shape
    psi = agg.patch_start_idx
    tokens = torch.cat([_special_tokens(agg.camera_token, B, S, has_reference),
                        _special_tokens(agg.register_token, B, S, has_reference), x], 1)
    pos = None
    if agg.rope is not None:
        pos = agg.position_getter(B * S, H // agg.patch_size, W // agg.patch_size, device=images.device) + 1
        pos = torch.cat([torch.zeros(B * S, psi, 2, device=images.device, dtype=pos.dtype), pos], 1)
    return tokens, pos, psi, P, C


def cached_attention_block(block, x_q, x_kv, pos_q, pos_kv):
    """A VGGT global block whose queries come from x_q and whose keys and values come
    from x_kv. With x_kv = x_q it equals the block's own forward pass."""
    attn = block.attn
    hq, hk = block.norm1(x_q), block.norm1(x_kv)
    B, Nq, C = hq.shape
    Nk = hk.shape[1]
    w, b = attn.qkv.weight, attn.qkv.bias
    q = F.linear(hq, w[:C], b[:C] if b is not None else None).reshape(B, Nq, attn.num_heads, -1).permute(0, 2, 1, 3)
    kv = F.linear(hk, w[C:], b[C:] if b is not None else None).reshape(B, Nk, 2, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = attn.q_norm(q), attn.k_norm(kv[0]), kv[1]
    del hq, hk
    if attn.rope is not None:
        q, k = attn.rope(q, pos_q), attn.rope(k, pos_kv)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C)
    x = x_q + block.drop_path1(block.ls1(attn.proj_drop(attn.proj(o))))
    return x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))


@torch.no_grad()
def pin_pass(model, images, score=False, score_queries=NUM_QUERIES):
    """Run the pinned frames [B, n, 3, H, W] (frame 0 first) with full attention among them.

    Returns the Scene Cache {global block: (input tokens, positions)}, the outputs of the
    cached blocks, the patch start index and, if requested, the Alignment Score.
    """
    agg = model.aggregator
    B, n = images.shape[:2]
    tokens, pos, psi, P, C = _embed(model, images, has_reference=True)
    T = psi + P
    cache, outs, align = {}, {}, None
    fi = gi = 0
    for blk_i in range(agg.aa_block_num):
        inter = {}
        for attn_type in agg.aa_order:
            if attn_type == "frame":
                tokens = agg.frame_blocks[fi](tokens.view(B * n, T, C), pos=(pos.view(B * n, T, 2) if pos is not None else None))
                fi += 1
                inter["frame"] = tokens.view(B, n, T, C)
            else:
                t = tokens.view(B, n * T, C)
                p = pos.view(B, n * T, 2) if pos is not None else None
                cache[blk_i] = (t, p)
                if score and blk_i == SCORE_BLOCK:
                    align = attention_divergence(agg.global_blocks[gi], tokens.view(B, n, T, C),
                                                 pos.view(B, n, T, 2) if pos is not None else None, psi, score_queries)
                out = cached_attention_block(agg.global_blocks[gi], t, t, p, p)
                gi += 1
                tokens = out.reshape(B * n, T, C)
                inter["global"] = tokens.view(B, n, T, C)
        if blk_i in CACHED_BLOCKS:
            outs[blk_i] = torch.cat([inter["frame"], inter["global"]], -1)
    return cache, outs, psi, align


@torch.no_grad()
def attach_pass(model, images, cache):
    """Run a batch of non-pinned frames [B, c, 3, H, W] whose global blocks read the Scene Cache."""
    agg = model.aggregator
    B, c = images.shape[:2]
    tokens, pos, psi, P, C = _embed(model, images, has_reference=False)
    T = psi + P
    outs = {}
    fi = gi = 0
    for blk_i in range(agg.aa_block_num):
        inter = {}
        for attn_type in agg.aa_order:
            if attn_type == "frame":
                tokens = agg.frame_blocks[fi](tokens.view(B * c, T, C), pos=(pos.view(B * c, T, 2) if pos is not None else None))
                fi += 1
                inter["frame"] = tokens.view(B, c, T, C)
            else:
                t = tokens.view(B, c * T, C)
                p = pos.view(B, c * T, 2) if pos is not None else None
                tk, pk = cache[blk_i]
                t = cached_attention_block(agg.global_blocks[gi], t, tk, p, pk)
                gi += 1
                tokens = t.reshape(B * c, T, C)
                inter["global"] = tokens.view(B, c, T, C)
        if blk_i in CACHED_BLOCKS:
            outs[blk_i] = torch.cat([inter["frame"], inter["global"]], -1)
    return outs


def head_inputs(model, outs, dtype):
    """Arrange the cached block outputs the way the depth head of the installed VGGT expects:
    a list of the four cached blocks (FastVGGT) or a list over all 24 blocks (official VGGT)."""
    layers = [outs[b].to(dtype) for b in CACHED_BLOCKS]
    head = getattr(model, "depth_head", None) or getattr(model, "point_head", None)
    idx = list(getattr(head, "intermediate_layer_idx", [0, 1, 2, 3])) if head is not None else [0, 1, 2, 3]
    if idx == [0, 1, 2, 3]:
        return layers
    full = [None] * (max(CACHED_BLOCKS) + 1)
    for b, x in zip(CACHED_BLOCKS, layers):
        full[b] = x
    return full


@torch.no_grad()
def reconstruct(model, images, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None, chunk: int = 64,
                tasks=("camera", "depth"), score: bool = False, amp_dtype=torch.bfloat16) -> Dict:
    """Reconstruct a video with EcoVGGT on a VGGT backbone.

    Args:
        model: a VGGT model (optionally with Alignment Refocusing weights loaded).
        images: [1, S, 3, H, W] in [0, 1], preprocessed like VGGT (width 518).
        num_keys: size of the Pinned Key Set (31 in the paper).
        key_frames: explicit pinned frame indices (frame 0 is added if missing).
        chunk: number of frames per attach batch.
        tasks: subset of ("camera", "depth", "point").
        score: also return the Alignment Score of the pin pass.
    Returns:
        dict with pose_enc [1, S, 9], depth [1, S, H, W, 1], depth_conf [1, S, H, W]
        (and world_points / world_points_conf for "point"), pinned_frames, alignment_score.
    """
    B, S = images.shape[:2]
    if B != 1:
        raise ValueError("EcoVGGT processes one video at a time (batch size 1)")
    dtype = next(model.parameters()).dtype
    keys = sorted(set([0] + [int(k) for k in (key_frames if key_frames is not None else pinned_keys(S, num_keys))]))
    rest = [i for i in range(S) if i not in set(keys)]
    use_amp = images.is_cuda and amp_dtype is not None
    amp = torch.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
    head_ctx = torch.autocast("cuda", enabled=False) if (dtype == torch.float32 and use_amp) else contextlib.nullcontext()
    cam_tok, depth, conf, wpts, wconf = [None] * S, [None] * S, [None] * S, [None] * S, [None] * S

    def heads(outs, frames, imgs, psi):
        layers = head_inputs(model, outs, dtype)
        last = [x for x in layers if x is not None][-1]
        with head_ctx:
            if "depth" in tasks:
                d, dc = model.depth_head(layers, images=imgs.to(dtype), patch_start_idx=psi)
                for j, f in enumerate(frames):
                    depth[f], conf[f] = d[:, j].cpu(), dc[:, j].cpu()
            if "point" in tasks and getattr(model, "point_head", None) is not None:
                p3, pc = model.point_head(layers, images=imgs.to(dtype), patch_start_idx=psi)
                for j, f in enumerate(frames):
                    wpts[f], wconf[f] = p3[:, j].cpu(), pc[:, j].cpu()
        for j, f in enumerate(frames):
            cam_tok[f] = last[:, j, :1].clone()

    with amp:
        cache, outs_k, psi, align = pin_pass(model, images[:, keys], score=score)
        heads(outs_k, keys, images[:, keys], psi)
        del outs_k
        for fr in batches(rest, chunk):
            outs = attach_pass(model, images[:, fr], cache)
            heads(outs, fr, images[:, fr], psi)
            del outs
        pred = {"pinned_frames": keys, "alignment_score": (align["c90"] if align else None)}
        if "camera" in tasks:
            with head_ctx:
                pose_enc_list = model.camera_head([torch.stack(cam_tok, 1)])
            pred["pose_enc"] = pose_enc_list[-1]
    if "depth" in tasks:
        pred["depth"], pred["depth_conf"] = torch.stack(depth, 1), torch.stack(conf, 1)
    if "point" in tasks and wpts[0] is not None:
        pred["world_points"], pred["world_points_conf"] = torch.stack(wpts, 1), torch.stack(wconf, 1)
    return pred
