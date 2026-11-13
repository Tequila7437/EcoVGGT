"""Small geometry helpers."""
import numpy as np
from scipy.linalg import svd


def umeyama_alignment(src, dst, estimate_scale=True):
    """Similarity (or rigid) transform that maps src [3, N] onto dst [3, N] in the least-squares
    sense (Umeyama, 1991). Returns (scale, R [3, 3], t [3])."""
    assert src.shape == dst.shape and src.shape[0] == 3
    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)
    src_c, dst_c = src - src_mean, dst - dst_mean
    try:
        U, D, Vt = svd(dst_c @ src_c.T)
        V = Vt.T
        S = np.eye(3)
        if np.linalg.det(U @ V.T) < 0:
            S[2, 2] = -1
        R = U @ S @ V.T
        if estimate_scale:
            var = np.sum(src_c * src_c)
            scale = 1.0 if var < 1e-10 else np.sum(D * np.diag(S)) / var
        else:
            scale = 1.0
        t = dst_mean.ravel() - scale * (R @ src_mean).ravel()
        return scale, R, t
    except Exception:
        return 1.0, np.eye(3), (dst_mean - src_mean).ravel()
