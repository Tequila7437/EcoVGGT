"""ScanNet metrics.

Trajectories follow the FastVGGT evaluation (see NOTICE.md): world-to-camera translations
are aligned with Sim(3), the first pose is then moved onto the first ground-truth pose, and
ATE, RPE-r, and RPE-t are RMS errors, the RPE between consecutive frames. For points, each
pixel of the central 224 x 224 crop is paired with the ground-truth point of the same pixel,
the prediction is aligned by Sim(3) and ICP, and accuracy and completeness are mean
nearest-neighbor distances. AbsRel scales each depth map by its median ratio to the ground
truth.
"""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from ecovggt.geometry import umeyama_alignment


def _trajectory(poses, frame_ids):
    from evo.core.trajectory import PoseTrajectory3D
    return PoseTrajectory3D(positions_xyz=poses[:, :3, 3],
                            orientations_quat_wxyz=Rotation.from_matrix(poses[:, :3, :3]).as_quat(scalar_first=True),
                            timestamps=np.asarray(frame_ids))


def _aligned(ref, est):
    """Sim(3) alignment, then origin alignment, as evo 1.31 and earlier did inside ape and rpe
    (newer versions refuse the two flags together)."""
    est = deepcopy(est)
    est.align(ref, correct_scale=True, correct_only_scale=False)
    est.align_origin(ref)
    return est


def trajectory_errors(w2c_pred, w2c_gt, frame_ids):
    """w2c_pred, w2c_gt: [S, 4, 4] world-to-camera transforms. Returns ATE (m), ARE (deg),
    RPE-r (deg), and RPE-t (m)."""
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core.metrics import PoseRelation, Unit
    ref, est = _trajectory(w2c_gt, frame_ids), _trajectory(w2c_pred, frame_ids)
    out = {}
    for name, rel in (("ate", PoseRelation.translation_part), ("are", PoseRelation.rotation_angle_deg)):
        out[name] = main_ape.ape(deepcopy(ref), _aligned(ref, est), est_name="traj", pose_relation=rel).stats["rmse"]
    for name, rel in (("rpe_rot", PoseRelation.rotation_angle_deg), ("rpe_trans", PoseRelation.translation_part)):
        out[name] = main_rpe.rpe(deepcopy(ref), _aligned(ref, est), est_name="traj", pose_relation=rel, delta=1,
                                 delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True).stats["rmse"]
    return {k: float(v) for k, v in out.items()}


def _nearest(gt, H, W):
    ys = (np.arange(H) * gt.shape[1] / H).astype(int)
    xs = (np.arange(W) * gt.shape[2] / W).astype(int)
    return gt[:, ys][:, :, xs]


def depth_errors(pred, gt):
    """pred [S, H, W] or [S, H, W, 1]; gt [S, h, w] in meters (0 = invalid), resampled to the
    prediction by nearest neighbors. Each predicted map is scaled by its median ratio to the
    ground truth. Returns AbsRel and the share of pixels with ratio below 1.25 (delta1)."""
    pred = pred[..., 0] if pred.ndim == 4 else pred
    S, H, W = pred.shape
    g = _nearest(gt, H, W)
    m = (g > 0.1) & (g < 10) & (pred > 1e-6)
    with np.errstate(all="ignore"):
        ratios = np.where(m, g / np.maximum(pred, 1e-6), np.nan)
        s = np.nanmedian(ratios.reshape(S, -1), axis=1)
        s = np.where(np.isfinite(s), s, 1.0)
    p = pred * s[:, None, None]
    err = np.abs(p - g) / np.maximum(g, 1e-6)
    d1 = np.maximum(p / np.maximum(g, 1e-6), g / np.maximum(p, 1e-6)) < 1.25
    return {"absrel": float(err[m].mean()), "delta1": float(d1[m].mean())}


def point_errors(pred_points, gt_depth, K_depth, depth_size, c2w_gt, max_points=999_999, seed=33):
    """Accuracy and completeness (m) of predicted world points [S, H, W, 3] (NaN = dropped)
    against the ground-truth points that the depth maps and poses lift from the same pixels."""
    import open3d as o3d
    from scipy.spatial import KDTree
    S, H, W = pred_points.shape[:3]
    fx, fy, cx, cy = K_depth
    sx, sy = W / depth_size[0], H / depth_size[1]
    fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
    g = _nearest(gt_depth, H, W)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    cam = np.stack([(u - cx) / fx * g, (v - cy) / fy * g, g], -1).astype(np.float64)
    gt_w = np.einsum("sij,shwj->shwi", c2w_gt[:, :3, :3], cam) + c2w_gt[:, None, None, :3, 3]
    yc, xc = H // 2, W // 2
    sl = (slice(None), slice(yc - 112, yc + 112), slice(xc - 112, xc + 112))
    pw, gw = pred_points[sl].astype(np.float64), gt_w[sl]
    m = (g[sl] > 0.1) & (g[sl] < 10) & np.isfinite(pw).all(-1)
    pw, gw = pw[m], gw[m]
    if len(pw) > max_points:  # same indices for both clouds, so pairs stay matched
        idx = np.random.default_rng(seed).choice(len(pw), max_points, replace=False)
        pw, gw = pw[idx], gw[idx]
    s, R, t = umeyama_alignment(pw.T, gw.T, True)
    pw = (s * (R @ pw.T)).T + t
    pa, pb = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pa.points, pb.points = o3d.utility.Vector3dVector(pw), o3d.utility.Vector3dVector(gw)
    reg = o3d.pipelines.registration.registration_icp(
        pa, pb, 0.1, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pw = np.asarray(pa.transform(reg.transformation).points)
    d_acc = KDTree(gw).query(pw, workers=-1)[0]
    d_comp = KDTree(pw).query(gw, workers=-1)[0]
    return {"acc": float(d_acc.mean()), "comp": float(d_comp.mean()), "num_points": int(len(pw))}
