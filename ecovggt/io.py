"""Reading frames and writing reconstructions."""
import glob
import json
import os
import re
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _natural_key(path: str):
    """Sort key that orders 2.jpg before 10.jpg."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(path))]


def list_frames(source: str, stride: int = 1, max_frames: int = 0) -> List:
    """Frames of a folder of images (in natural name order) or of a video file.
    Returns file paths for folders and RGB arrays for videos."""
    if os.path.isdir(source):
        frames = sorted((p for p in glob.glob(os.path.join(source, "*")) if p.lower().endswith(IMAGE_EXTS)), key=_natural_key)
        if not frames:
            raise FileNotFoundError(f"no images with extensions {IMAGE_EXTS} in {source}")
    else:
        cap, frames, i = cv2.VideoCapture(source), [], 0
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {source}")
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        stride = 1
    frames = frames[::stride]
    return frames[:max_frames] if max_frames else frames


def preprocess(frame, width: int = 518, patch: int = 14) -> torch.Tensor:
    """VGGT-style preprocessing: resize to `width` with bicubic interpolation, height rounded
    to a multiple of `patch` and center-cropped to at most `width`. Returns [3, H, W] in [0, 1].
    Images are decoded with OpenCV, as in the evaluation."""
    if isinstance(frame, str):
        arr = cv2.imread(frame)
        if arr is None:
            raise FileNotFoundError(frame)
        frame = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.ascontiguousarray(frame[..., :3]))
    w, h = img.size
    new_h = round(h * (width / w) / patch) * patch
    img = img.resize((width, new_h), Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
    if new_h > width:
        top = (new_h - width) // 2
        x = x[:, top:top + width]
    return x


def load_images(source: str, stride: int = 1, max_frames: int = 0, backbone: str = "vggt") -> torch.Tensor:
    """[S, 3, H, W] in [0, 1] from a folder of images or a video, preprocessed for the backbone:
    518 pixels wide for VGGT, pi3 and DA3, and the official 416-pixel "balanced" resizing of
    VGGT-Omega (about 480 x 352 for 4:3 frames)."""
    frames = list_frames(source, stride, max_frames)
    if backbone != "vggt-omega":
        return torch.stack([preprocess(f) for f in frames])
    from .models import _add_path
    _add_path("ECOVGGT_VGGT_OMEGA_ROOT", "vggt-omega")
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    if isinstance(frames[0], str):
        return load_and_preprocess_images(frames, mode="balanced", image_resolution=416)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:  # video frames: lossless files for the official loader
        paths = []
        for i, f in enumerate(frames):
            paths.append(os.path.join(tmp, f"{i:06d}.png"))
            Image.fromarray(f).save(paths[-1])
        return load_and_preprocess_images(paths, mode="balanced", image_resolution=416)


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Binary little-endian PLY with float32 xyz and uint8 rgb."""
    points = points.astype(np.float32)
    colors = colors.astype(np.uint8)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
        rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
        f.write(rec.tobytes())


def save_reconstruction(out_dir: str, rec: dict, images: torch.Tensor, conf_percentile: float = 50.0, max_points: int = 2_000_000):
    """Write cameras (w2c and intrinsics per frame, JSON), depth maps (NPZ) and a colored
    point cloud (PLY) of the pixels whose confidence is above the given percentile."""
    os.makedirs(out_dir, exist_ok=True)
    S = rec["w2c"].shape[0]
    cams = [{"frame": i, "w2c": rec["w2c"][i].tolist(), "intrinsics": rec["intrinsics"][i].tolist()} for i in range(S)]
    meta = {"pinned_frames": [int(k) for k in rec["pinned_frames"]], "alignment_score": rec.get("alignment_score"), "cameras": cams}
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(meta, f)
    np.savez_compressed(os.path.join(out_dir, "depth.npz"), depth=rec["depth"].astype(np.float16), conf=rec["conf"].astype(np.float16))
    conf = rec["conf"]
    keep = (conf >= np.percentile(conf, conf_percentile)) & np.isfinite(rec["world_points"]).all(-1)
    rgb = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pts, cols = rec["world_points"][keep], rgb[keep]
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    write_ply(os.path.join(out_dir, "points.ply"), pts, cols)
    return len(pts)
