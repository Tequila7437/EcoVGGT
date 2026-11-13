"""Evaluate EcoVGGT on long ScanNet videos with the protocol of the paper.

Each video keeps its first frame and then takes frames at the largest integer stride that
fills the frame budget (the protocol of FastVGGT). A scene succeeds when its ATE is below
0.2 m. The script writes one JSON file per scene and a summary with the success rate and
the medians over scenes.

Example (EcoVGGT (VGGT) offline on 500-frame videos):
    python eval/eval_scannet.py --data_dir /data/scannet/process_scannet --sens_root /data/scannet/scans \
        --backbone vggt --num_frames 500 --out results/vggt_500
Add --online for the Online Stream (key interval 16 up to 500 frames and 32 above, as in the paper).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ecovggt  # noqa: E402
from ecovggt import da3_engine, online, pi3_engine, vggt_engine, vggt_omega_engine  # noqa: E402
from ecovggt.io import preprocess  # noqa: E402
from eval.metrics import depth_errors, point_errors, trajectory_errors  # noqa: E402
from eval.prepare_scannet import find_sens, read_sens_depth  # noqa: E402
from ecovggt.models import _add_path  # noqa: E402

_add_path("ECOVGGT_VGGT_ROOT", "FastVGGT")  # the protocol uses the camera encoding and unprojection of VGGT for every backbone

SUCCESS_ATE = 0.2  # meters


# ----------------------------------------------------------------------------- data
def load_poses(pose_dir):
    """Valid camera-to-world poses in numerical frame order, expressed relative to the first."""
    files = sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem))
    c2w, ids = [], []
    for p in files:
        m = np.array([float(x) for x in p.read_text().split()]).reshape(4, 4)
        if np.isfinite(m).all():
            c2w.append(m)
            ids.append(int(p.stem))
    c2w = np.stack(c2w)
    return np.linalg.inv(c2w[0]) @ c2w, np.array(ids)


def select_frames(image_ids, pose_ids, budget):
    """First frame, then every `step`-th frame with a pose, `step` the largest integer stride
    that fills the budget."""
    valid = sorted(set(image_ids) & set(pose_ids.tolist()))
    if len(valid) <= budget:
        return valid
    rest = valid[1:]
    step = max(1, len(rest) // (budget - 1))
    return [valid[0]] + rest[::step][:budget - 1]


def load_scene(scene_dir, budget):
    images = {int(p.stem): p for ext in ("*.jpg", "*.png", "*.jpeg") for p in (scene_dir / "color").glob(ext)}
    c2w, pose_ids = load_poses(scene_dir / "pose")
    ids = select_frames(list(images), pose_ids, budget)
    index = {f: i for i, f in enumerate(pose_ids)}
    paths = [str(images[f]) for f in ids]
    x = torch.stack([preprocess(p) for p in paths])
    return x, ids, c2w[[index[f] for f in ids]], paths


# ----------------------------------------------------------------------------- inference
def load_vggt_omega_frames(paths):
    """The original frames with the official 416-pixel preprocessing of VGGT-Omega."""
    _add_path("ECOVGGT_VGGT_OMEGA_ROOT", "vggt-omega")
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    return load_and_preprocess_images(paths, mode="balanced", image_resolution=416).pin_memory()


def predict_vggt_omega(model, imgs, H, W, online_mode, key_interval):
    """VGGT-Omega runs on its own preprocessing of the frames (imgs). Its depth and confidence
    are resized bilinearly to the H x W grid of the evaluation, and its focal lengths are
    scaled to this grid with the principal point at the center."""
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding
    from vggt_omega.utils.pose_enc import encoding_to_camera
    S, _, h, w = imgs.shape
    if online_mode:
        res = vggt_omega_engine.online_reconstruct(model, imgs[None], key_interval=key_interval, lag=2, max_keys=32, chunk=64)
    else:
        res = vggt_omega_engine.reconstruct_raw(model, imgs[None], num_keys=31, chunk=64)
    w2c, K = encoding_to_camera(res["pose_enc"].cuda(), (h, w))
    depth, conf = res["depth"].float().cpu(), res["depth_conf"].float().cpu()
    w2c, K = w2c.double().cpu(), K.double().cpu()
    w2c4 = torch.cat([w2c, torch.tensor([0., 0., 0., 1.], dtype=torch.float64).view(1, 1, 1, 4).expand(1, S, 1, 4)], -2)
    c2w = torch.linalg.inv(w2c4)
    w2c_rel = torch.linalg.inv(torch.linalg.inv(c2w[:, :1]) @ c2w)
    d = torch.nn.functional.interpolate(depth.reshape(S, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    c = torch.nn.functional.interpolate(conf.reshape(S, 1, h, w), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    Ke = K.clone()
    Ke[..., 0, 0] *= W / w
    Ke[..., 1, 1] *= H / h
    Ke[..., 0, 2], Ke[..., 1, 2] = W / 2.0, H / 2.0
    pose_enc = extri_intri_to_pose_encoding(w2c_rel[:, :, :3].float(), Ke.float(), (H, W))
    return pose_enc, d[None, ..., None].float(), c[None].float(), None, None


def predict(model, backbone, x, online_mode, key_interval, frames=None):
    """x [S, 3, H, W] bfloat16 on the GPU. Returns pose encodings [1, S, 9], depth
    [1, S, H, W, 1], and a confidence [1, S, H, W] whose threshold of 1 drops unreliable pixels.
    For pi3 and DA3, also returns their own world points [1, S, H, W, 3] with a confidence on
    the same scale; as in the paper, the point metrics of these backbones use these points,
    and those of VGGT and VGGT-Omega unproject the depth maps."""
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding
    S, _, H, W = x.shape
    if backbone == "vggt-omega":
        return predict_vggt_omega(model, frames, H, W, online_mode, key_interval)
    if backbone in ("vggt", "vggt-frozen"):
        if online_mode:
            res = online.online_reconstruct_vggt(model, x[None], key_interval=key_interval, lag=2, max_keys=32, chunk=64)
        else:
            res = vggt_engine.reconstruct(model, x[None], num_keys=31, chunk=64)
        dev = x.device if S <= 1500 else torch.device("cpu")
        return res["pose_enc"].to(dev), res["depth"].to(dev), res["depth_conf"].to(dev), None, None
    if backbone == "pi3":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if online_mode:
                raw = online.online_reconstruct_pi3(model, x.float()[None], key_interval=key_interval, max_keys=32, chunk=64)
            else:
                raw = pi3_engine.reconstruct_raw(model, x.float()[None], num_keys=31, chunk=64)
        out = pi3_engine.to_camera_outputs(raw, H, W, device=x.device)
        with torch.autocast("cuda", enabled=False):
            pose_enc = extri_intri_to_pose_encoding(out["w2c"][None], out["intrinsics"][None], (H, W))
        conf = (out["conf"] / 0.1)[None]
        return pose_enc, out["depth"][None, ..., None], conf, out["world_points"][None], conf
    if backbone == "da3":
        if online_mode:
            raise SystemExit("the Online Stream supports the vggt, pi3 and vggt-omega backbones")
        out = da3_engine.reconstruct(model, x.float()[None], num_keys=31, chunk=64)
        with torch.autocast("cuda", enabled=False):
            pose_enc = extri_intri_to_pose_encoding(out["w2c"][None], out["intrinsics"][None], (H, W))
        return pose_enc.cpu(), out["depth"][None, ..., None], out["conf"][None], out["world_points"][None], out["conf"][None]
    raise ValueError(backbone)


def evaluate_scene(model, args, scene):
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x, frame_ids, c2w_gt, paths = load_scene(Path(args.data_dir) / scene, args.num_frames)
    S, _, H, W = x.shape
    key_interval = args.key_interval or (16 if args.num_frames <= 500 else 32)
    frames = load_vggt_omega_frames(paths) if args.backbone == "vggt-omega" else None  # decoding stays outside the timing
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        pose_enc, depth, conf, wpts, wconf = predict(model, args.backbone, x.cuda().to(torch.bfloat16), args.online, key_interval, frames)
    torch.cuda.synchronize()
    seconds = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    w2c, K = pose_encoding_to_extri_intri(pose_enc, (H, W))  # in the dtype of the camera head, as in the paper
    w2c, K = w2c[0].float().cpu().numpy(), K[0].float().cpu().numpy()
    depth = depth[0].float().cpu().numpy()
    raw_depth = depth.copy()
    depth[conf[0].float().cpu().numpy() < 1.0] = np.nan
    if wpts is not None:
        points = wpts[0].float().cpu().numpy()
        points[wconf[0].float().cpu().numpy() < 1.0] = np.nan
    else:
        points = unproject_depth_map_to_point_map(depth, w2c, K)
    w2c_pred = np.tile(np.eye(4), (S, 1, 1))
    w2c_pred[:, :3, :4] = w2c
    metrics = trajectory_errors(w2c_pred, np.linalg.inv(c2w_gt), frame_ids)
    if args.sens_root:
        gt_depth, K_depth, depth_size = read_sens_depth(find_sens(args.sens_root, scene), frame_ids)
        metrics.update(depth_errors(raw_depth, gt_depth))
        metrics.update(point_errors(points, gt_depth, K_depth, depth_size, c2w_gt.astype(np.float64)))
    metrics.update(num_frames=S, seconds=seconds, peak_memory_gb=peak_gb, key_interval=key_interval if args.online else None)
    return metrics


# ----------------------------------------------------------------------------- main
def summarize(per_scene):
    done = [m for m in per_scene.values() if "ate" in m]
    med = lambda k: float(np.median([m[k] for m in done if m.get(k) is not None])) if any(k in m for m in done) else None
    return {"num_scenes": len(per_scene),
            "success_rate": 100.0 * sum(m.get("ate", np.inf) < SUCCESS_ATE for m in per_scene.values()) / len(per_scene),
            **{f"median_{k}": med(k) for k in ("ate", "rpe_rot", "rpe_trans", "acc", "comp", "absrel", "seconds", "peak_memory_gb")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="exported scenes (eval/prepare_scannet.py)")
    ap.add_argument("--sens_root", default=None, help="ScanNet .sens files for the depth and point metrics (omit for trajectories only)")
    ap.add_argument("--scenes", default=str(Path(__file__).with_name("scannet_50.txt")))
    ap.add_argument("--backbone", default="vggt", choices=["vggt", "vggt-frozen", "pi3", "da3", "vggt-omega"])
    ap.add_argument("--num_frames", type=int, default=500)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--key_interval", type=int, default=0, help="Online Stream key interval (default: 16 up to 500 frames, else 32)")
    ap.add_argument("--weights", default=None, help="local backbone weights (see demo.py)")
    ap.add_argument("--refocusing_weights", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kwargs = {k: v for k, v in (("weights", a.weights), ("refocusing_weights", a.refocusing_weights if a.backbone == "vggt" else None)) if v}
    model = ecovggt.load_backbone(a.backbone, **kwargs)
    os.makedirs(a.out, exist_ok=True)
    per_scene = {}
    for scene in [l.strip() for l in open(a.scenes) if l.strip()]:
        path = os.path.join(a.out, f"{scene}.json")
        if os.path.exists(path):
            per_scene[scene] = json.load(open(path))
            continue
        try:
            m = evaluate_scene(model, a, scene)
        except torch.cuda.OutOfMemoryError:
            m = {"error": "out of memory"}
            torch.cuda.empty_cache()
        json.dump(m, open(path, "w"), indent=2)
        per_scene[scene] = m
        print(f"{scene}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)), flush=True)
    s = summarize(per_scene)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
