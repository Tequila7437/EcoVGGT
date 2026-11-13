"""ecovggt.reconstruct: one call for every backbone."""
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from . import da3_engine, pi3_engine, vggt_engine, vggt_omega_engine


def _unproject(depth, w2c, K):
    """depth [S, H, W], w2c [S, 3, 4], K [S, 3, 3] (numpy) -> world points [S, H, W, 3]."""
    S, H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    pts = np.empty((S, H, W, 3), dtype=np.float32)
    for i in range(S):
        z = depth[i].astype(np.float64)
        x = (u - K[i, 0, 2]) / K[i, 0, 0] * z
        y = (v - K[i, 1, 2]) / K[i, 1, 1] * z
        cam = np.stack([x, y, z], -1)
        R, t = w2c[i, :3, :3].astype(np.float64), w2c[i, :3, 3].astype(np.float64)
        pts[i] = ((cam - t) @ R).astype(np.float32)  # X_world = R^T (X_cam - t)
    return pts


@torch.no_grad()
def reconstruct(model, images, backbone: str, num_keys: int = 31, key_frames: Optional[Sequence[int]] = None,
                chunk: int = 64, score: bool = False) -> Dict:
    """Reconstruct a video with EcoVGGT.

    Args:
        model: a backbone from ecovggt.load_backbone.
        images: [S, 3, H, W] or [1, S, 3, H, W] in [0, 1] (see ecovggt.io.load_images).
        backbone: "vggt", "vggt-frozen", "pi3", "da3" or "vggt-omega".
        num_keys: size of the Pinned Key Set; key_frames overrides it.
        chunk: frames per attach batch (memory scales with chunk + num_keys, not with S).
        score: also compute the Alignment Score (VGGT backbones).
    Returns:
        numpy arrays in the camera frame of frame 0: w2c [S, 3, 4] (OpenCV), intrinsics
        [S, 3, 3], depth [S, H, W], conf [S, H, W], world_points [S, H, W, 3]; plus
        pinned_frames and alignment_score.
    """
    if images.dim() == 4:
        images = images[None]
    H, W = images.shape[-2:]
    if backbone == "vggt-omega":
        out = vggt_omega_engine.reconstruct(model, images, num_keys=num_keys, key_frames=key_frames, chunk=chunk)
        return dict(w2c=out["w2c"].numpy(), intrinsics=out["intrinsics"].numpy(), depth=out["depth"].numpy(), conf=out["conf"].numpy(),
                    world_points=out["world_points"].numpy(), pinned_frames=out["pinned_frames"], alignment_score=None)
    if backbone in ("vggt", "vggt-frozen"):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        dtype = next(model.parameters()).dtype
        pred = vggt_engine.reconstruct(model, images.to(dtype), num_keys=num_keys, key_frames=key_frames, chunk=chunk, score=score)
        w2c, K = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (H, W))
        w2c, K = w2c[0].cpu().numpy(), K[0].cpu().numpy()
        depth = pred["depth"][0, ..., 0].float().numpy()
        conf = pred["depth_conf"][0].float().numpy()
        return dict(w2c=w2c, intrinsics=K, depth=depth, conf=conf, world_points=_unproject(depth, w2c, K),
                    pinned_frames=pred["pinned_frames"], alignment_score=pred["alignment_score"])
    engine = {"pi3": pi3_engine, "da3": da3_engine}.get(backbone)
    if engine is None:
        raise ValueError(f"unknown backbone {backbone!r}")
    out = engine.reconstruct(model, images, num_keys=num_keys, key_frames=key_frames, chunk=chunk)
    return dict(w2c=out["w2c"].cpu().numpy(), intrinsics=out["intrinsics"].cpu().numpy(), depth=out["depth"].float().cpu().numpy(),
                conf=out["conf"].float().cpu().numpy(), world_points=out["world_points"].float().cpu().numpy(),
                pinned_frames=out["pinned_frames"], alignment_score=None)
