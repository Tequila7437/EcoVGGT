r"""Export ScanNet .sens files for evaluation and training.

Evaluation layout (all frames, FastVGGT protocol):
    <out>/<scene>/color/<i>.jpg      original JPEG bytes of frame i
    <out>/<scene>/pose/<i>.txt       camera-to-world 4x4 (invalid poses written as -inf)
    <out>/<scene>/intrinsic/*.txt    color and depth intrinsics and extrinsics
The evaluation reads ground-truth depth from the .sens files directly.

Training layout (--train, every 5th frame with a valid pose) adds
    <out>/<scene>/depth/<i>.png      16-bit depth in millimeters

    python eval/prepare_scannet.py --sens_root /data/scannet/scans --scenes eval/scannet_50.txt \
        --out /data/scannet/process_scannet
    python eval/prepare_scannet.py --sens_root /data/scannet/scans --scenes training/scannet_train_195.txt \
        --out /data/scannet/process_scannet_train --train
"""
import argparse
import os
import struct
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

COLOR_JPEG = 2
DEPTH_ZLIB_USHORT = 1


def _mat_lines(vals):
    return "".join(" ".join("%f" % v for v in vals[4 * i:4 * i + 4]) + "\n" for i in range(4))


def _read_header(f):
    assert struct.unpack("I", f.read(4))[0] == 4, "only .sens version 4 is supported"
    f.read(struct.unpack("Q", f.read(8))[0])  # sensor name
    mats = {name: struct.unpack("f" * 16, f.read(64))
            for name in ("intrinsic_color", "extrinsic_color", "intrinsic_depth", "extrinsic_depth")}
    color_comp, depth_comp = struct.unpack("i", f.read(4))[0], struct.unpack("i", f.read(4))[0]
    cw, ch, dw, dh = struct.unpack("IIII", f.read(16))
    shift = struct.unpack("f", f.read(4))[0]
    num_frames = struct.unpack("Q", f.read(8))[0]
    assert color_comp == COLOR_JPEG and depth_comp == DEPTH_ZLIB_USHORT, f"compression {color_comp}/{depth_comp}"
    return mats, (cw, ch, dw, dh), shift, num_frames


def export_scene(sens, out, train=False, stride=5):
    """Export one scene; returns a one-line summary. Existing exports are kept."""
    done = os.path.join(out, "EXPORT_DONE")
    if os.path.isfile(done):
        return f"{out}: exists"
    for d in ("color", "pose", "intrinsic") + (("depth",) if train else ()):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with open(sens, "rb") as f:
        mats, (cw, ch, dw, dh), shift, num_frames = _read_header(f)
        for name, m in mats.items():
            with open(os.path.join(out, "intrinsic", f"{name}.txt"), "w") as g:
                g.write(_mat_lines(m))
        written = 0
        for i in range(num_frames):
            pose = struct.unpack("f" * 16, f.read(64))
            f.read(16)  # timestamps
            csz, dsz = struct.unpack("QQ", f.read(16))
            valid = all(np.isfinite(pose))
            if train and (i % stride != 0 or not valid):
                f.seek(csz + dsz, 1)
                continue
            color = f.read(csz)
            depth = f.read(dsz)
            with open(os.path.join(out, "color", f"{i}.jpg"), "wb") as g:
                g.write(color)
            with open(os.path.join(out, "pose", f"{i}.txt"), "w") as g:
                g.write(_mat_lines(pose))
            if train:
                from PIL import Image
                d = np.frombuffer(zlib.decompress(depth), dtype=np.uint16).reshape(dh, dw)
                Image.fromarray((d.astype(np.float32) / shift * 1000.0).round().astype(np.uint16)).save(
                    os.path.join(out, "depth", f"{i}.png"))
            written += 1
    with open(done, "w") as g:
        g.write(f"frames={written} of {num_frames} color={cw}x{ch} depth={dw}x{dh} stride={stride if train else 1}\n")
    return f"{out}: {written} of {num_frames} frames"


def read_sens_depth(sens, frame_ids):
    """Depth maps (meters, float32 [n, dh, dw], 0 where invalid) of the given frames, sorted
    ascending, with the depth intrinsics (fx, fy, cx, cy) and the depth size (dw, dh)."""
    want, out = sorted(frame_ids), {}
    with open(sens, "rb") as f:
        mats, (_, _, dw, dh), shift, num_frames = _read_header(f)
        k = 0
        for i in range(num_frames):
            f.read(64 + 16)
            csz, dsz = struct.unpack("QQ", f.read(16))
            if k < len(want) and i == want[k]:
                f.read(csz)
                d = np.frombuffer(zlib.decompress(f.read(dsz)), dtype=np.uint16).reshape(dh, dw)
                out[i] = d.astype(np.float32) / shift
                k += 1
            else:
                f.seek(csz + dsz, 1)
            if k >= len(want):
                break
    K = np.array(mats["intrinsic_depth"]).reshape(4, 4)
    return np.stack([out[i] for i in want]), (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), (dw, dh)


def find_sens(sens_root, scene):
    for p in (os.path.join(sens_root, scene, f"{scene}.sens"), os.path.join(sens_root, f"{scene}.sens")):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{scene}.sens not found under {sens_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sens_root", required=True, help="folder with <scene>/<scene>.sens (the layout of the ScanNet download)")
    ap.add_argument("--scenes", required=True, help="text file with one scene name per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", action="store_true", help="training layout: every 5th frame with a valid pose, plus depth")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    scenes = [l.strip() for l in open(a.scenes) if l.strip()]
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = [ex.submit(export_scene, find_sens(a.sens_root, s), os.path.join(a.out, s), a.train) for s in scenes]
        for j in jobs:
            print(j.result(), flush=True)


if __name__ == "__main__":
    main()
