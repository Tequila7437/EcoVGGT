"""Choice of the Pinned Key Set."""
from typing import List


def pinned_keys(num_frames: int, num_keys: int = 31) -> List[int]:
    """Indices of `num_keys` frames spread evenly over a video of `num_frames` frames.

    The first and the last frame are always pinned. Frame 0 is the reference frame
    of VGGT, so every reconstruction is expressed in its camera coordinates.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    n = min(int(num_keys), num_frames)
    if n <= 1:
        return [0]
    return sorted(set([0] + [round(i * (num_frames - 1) / (n - 1)) for i in range(1, n)]))


def batches(frames: List[int], size: int):
    """Split `frames` into consecutive batches of at most `size` frames."""
    return [frames[s:s + size] for s in range(0, len(frames), size)]
