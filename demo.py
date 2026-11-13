"""Reconstruct a folder of images or a video with EcoVGGT.

Examples:
    python demo.py --input path/to/frames --backbone pi3 --out outputs/scene
    python demo.py --input video.mp4 --stride 5 --backbone vggt --score --out outputs/video
    python demo.py --input path/to/frames --backbone pi3 --online --out outputs/stream
"""
import argparse
import time

import numpy as np
import torch

import ecovggt
from ecovggt.io import load_images, save_reconstruction


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of images or a video file")
    ap.add_argument("--out", required=True, help="output folder (cameras.json, depth.npz, points.ply)")
    ap.add_argument("--backbone", default="pi3", choices=["vggt", "vggt-frozen", "pi3", "da3", "vggt-omega"])
    ap.add_argument("--weights", default=None, help="backbone weights: a local VGGT-1B or VGGT-Omega checkpoint, or a Hugging Face id or folder for pi3 and da3")
    ap.add_argument("--refocusing_weights", default=None, help="local Alignment Refocusing checkpoint (default: download)")
    ap.add_argument("--num_keys", type=int, default=31, help="size of the Pinned Key Set")
    ap.add_argument("--chunk", type=int, default=64, help="frames per attach batch")
    ap.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--online", action="store_true", help="process frames in arrival order with the Online Stream (all backbones except da3)")
    ap.add_argument("--key_interval", type=int, default=16, help="Online Stream: frames between candidate keys")
    ap.add_argument("--max_keys", type=int, default=32, help="Online Stream: key budget per pass")
    ap.add_argument("--score", action="store_true", help="report the Alignment Score (vggt backbones)")
    ap.add_argument("--conf_percentile", type=float, default=50.0, help="keep points above this confidence percentile")
    a = ap.parse_args()

    images = load_images(a.input, a.stride, a.max_frames, backbone=a.backbone)
    print(f"{images.shape[0]} frames of {images.shape[-1]}x{images.shape[-2]} from {a.input}")
    kwargs = {}
    if a.weights:
        kwargs["weights"] = a.weights
    if a.refocusing_weights and a.backbone == "vggt":
        kwargs["refocusing_weights"] = a.refocusing_weights
    model = ecovggt.load_backbone(a.backbone, **kwargs)
    t0 = time.time()
    if a.online:
        rec = online(model, images.cuda(), a)
    else:
        rec = ecovggt.reconstruct(model, images.cuda(), a.backbone, num_keys=a.num_keys, chunk=a.chunk, score=a.score)
    print(f"reconstructed in {time.time() - t0:.1f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    if rec.get("alignment_score") is not None:
        s = rec["alignment_score"]
        print(f"Alignment Score {s:.2f}" + (" (below threshold: the reconstruction may have failed)" if s < ecovggt.THRESHOLD_PINNED_KEYS else ""))
    n = save_reconstruction(a.out, rec, images, a.conf_percentile)
    print(f"wrote {n} points and {images.shape[0]} cameras to {a.out}")


def online(model, images, a):
    """Feed frames one by one to the Online Stream and convert its outputs."""
    from ecovggt import pi3_engine, vggt_omega_engine
    from ecovggt.api import _unproject
    H, W = images.shape[-2:]
    if a.backbone == "vggt-omega":
        raw = vggt_omega_engine.online_reconstruct(model, images[None], key_interval=a.key_interval, lag=2, max_keys=a.max_keys, chunk=a.chunk)
        out = vggt_omega_engine.to_camera_outputs(raw, H, W)
        return dict(w2c=out["w2c"].numpy(), intrinsics=out["intrinsics"].numpy(), depth=out["depth"].numpy(), conf=out["conf"].numpy(),
                    world_points=out["world_points"].numpy(), pinned_frames=raw["pinned_frames"])
    if a.backbone == "pi3":
        stream = ecovggt.Pi3OnlineStream(model, key_interval=a.key_interval, max_keys=a.max_keys, chunk=a.chunk)
    elif a.backbone in ("vggt", "vggt-frozen"):
        stream = ecovggt.VGGTOnlineStream(model, key_interval=a.key_interval, lag=2, max_keys=a.max_keys, chunk=a.chunk)
    else:
        raise SystemExit("the Online Stream supports the vggt, pi3 and vggt-omega backbones")
    dtype = next(model.parameters()).dtype
    for i in range(images.shape[0]):
        stream.push(images[i].to(dtype) if a.backbone in ("vggt", "vggt-frozen") else images[i])
    stream.finish()
    res = stream.results()
    if a.backbone == "pi3":
        out = pi3_engine.to_camera_outputs(res, H, W, device=images.device)
        return dict(w2c=out["w2c"].cpu().numpy(), intrinsics=out["intrinsics"].cpu().numpy(), depth=out["depth"].cpu().numpy(),
                    conf=out["conf"].cpu().numpy(), world_points=out["world_points"].cpu().numpy(), pinned_frames=stream.candidates)
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    w2c, K = pose_encoding_to_extri_intri(res["pose_enc"].float(), (H, W))
    w2c, K = w2c[0].cpu().numpy(), K[0].cpu().numpy()
    depth, conf = res["depth"][0, ..., 0].float().numpy(), res["depth_conf"][0].float().numpy()
    return dict(w2c=w2c, intrinsics=K, depth=depth, conf=conf, world_points=_unproject(depth, w2c, K), pinned_frames=stream.candidates)


if __name__ == "__main__":
    main()
