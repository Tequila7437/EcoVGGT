"""Alignment Score: how sharply the alignment band of VGGT attends during the pin pass.

For a query token, the divergence of its attention distribution p over N key tokens
from the uniform distribution u is KL(u || p) = logsumexp(l) - mean(l) - log N, where
l are the attention logits. The score is the 90th percentile of this divergence over
2,048 random patch queries of the pinned frames and all heads of global block 13.
A low score means the band failed to concentrate its attention, which flags a likely
failed reconstruction (see the paper and its supplementary material).
"""
import math

import torch
import torch.nn.functional as F

SCORE_BLOCK = 13
NUM_QUERIES = 2048
# Scores below these thresholds flag a likely failure. On 500-frame ScanNet videos, the
# best threshold is 7.83 under full attention and lies between 9.01 and 9.15 for key sets
# of a quarter down to a thirty-second of the frames, so the paper uses 9.09 for all
# restricted key sets. On longer videos, the best threshold depends on the length.
THRESHOLD_FULL_ATTENTION = 7.83
THRESHOLD_PINNED_KEYS = 9.09


@torch.no_grad()
def attention_divergence(block, tokens, pos, patch_start_idx, num_queries=NUM_QUERIES, seed=0):
    """Divergence statistics of one VGGT global block.

    Args:
        block: a VGGT global attention block.
        tokens: block input tokens of the pinned frames, [B, n, T, C].
        pos: their 2D positions for RoPE, [B, n, T, 2], or None.
        patch_start_idx: number of special tokens before the patch tokens.
    Returns:
        dict with the median (c50) and the 90th percentile (c90, the Alignment Score).
    """
    B, S, T, C = tokens.shape
    attn = block.attn
    dh = C // attn.num_heads
    x_kv = tokens.reshape(B, -1, C)
    p_kv = pos.reshape(B, -1, 2) if pos is not None else None
    g = torch.Generator(device="cpu").manual_seed(seed)
    qs = torch.randint(0, S, (num_queries,), generator=g)
    qt = torch.randint(patch_start_idx, T, (num_queries,), generator=g)
    x_q = tokens[:, qs, qt]
    p_q = pos[:, qs, qt] if pos is not None else None
    hq, hk = block.norm1(x_q), block.norm1(x_kv)
    w, b = attn.qkv.weight, attn.qkv.bias
    q = F.linear(hq, w[:C], b[:C] if b is not None else None).reshape(B, num_queries, attn.num_heads, dh).permute(0, 2, 1, 3)
    k = F.linear(hk, w[C:2 * C], b[C:2 * C] if b is not None else None).reshape(B, -1, attn.num_heads, dh).permute(0, 2, 1, 3)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None:
        q, k = attn.rope(q, p_q), attn.rope(k, p_kv)
    n_keys = k.shape[2]
    parts = []
    for i in range(0, num_queries, 32):
        logits = torch.einsum("bhnd,bhmd->bhnm", q[:, :, i:i + 32].float(), k.float()) * attn.scale
        parts.append(torch.logsumexp(logits, -1) - logits.mean(-1) - math.log(n_keys))
        del logits
    c = torch.cat(parts, 2).reshape(-1)
    return {"c50": float(c.quantile(0.5)), "c90": float(c.quantile(0.9)), "num_keys": int(n_keys)}
