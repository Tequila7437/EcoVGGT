"""Loading the backbones and the Alignment Refocusing weights."""
import inspect
import os
import sys

import torch

# Paper setting: the VGGT implementation of FastVGGT with bfloat16 weights. Point
# ECOVGGT_VGGT_ROOT (or third_party/FastVGGT next to this repository) at a FastVGGT checkout;
# without it, the official `vggt` package is used with float32 weights and bfloat16 autocast.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFOCUSING_REPO = os.environ.get("ECOVGGT_HF_REPO", "Tequila7437/EcoVGGT")
REFOCUSING_FILE = "alignment_refocusing_vggt1b.safetensors"
# Hugging Face revisions of the backbone weights used in the paper
REVISIONS = {
    "facebook/VGGT-1B": "860abec7937da0a4c03c41d3c269c366e82abdf9",
    "yyfz233/Pi3": "ae722e7039287d0c8fde9f11f197f804f44b510c",
    "depth-anything/DA3-LARGE": "c54c26b16ec04d218e8d584ecf4bce082a9fcc20",
    "facebook/VGGT-Omega": "1041e80fc0e911235d3426b0a3d9a81075111a53",
}


def _add_path(env, default):
    root = os.environ.get(env) or os.path.join(_REPO, "third_party", default)
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
    return root


def _download(repo_id, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=REVISIONS.get(repo_id))


def load_refocusing(model, path=None):
    """Load Alignment Refocusing (fine-tuned global blocks 10 to 15) into a VGGT model."""
    from safetensors.torch import load_file
    path = path or _download(REFOCUSING_REPO, REFOCUSING_FILE)
    sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
    dtype = next(model.parameters()).dtype
    missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {path}: {unexpected[:5]}")
    loaded = set(sd)
    if not all(k.startswith("aggregator.global_blocks.") for k in loaded):
        raise RuntimeError(f"{path} does not look like an Alignment Refocusing checkpoint")
    return model


def load_vggt(refocusing=True, weights=None, refocusing_weights=None, device="cuda", point_head=False):
    """VGGT-1B, optionally with Alignment Refocusing (the paper's EcoVGGT (VGGT))."""
    _add_path("ECOVGGT_VGGT_ROOT", "FastVGGT")
    from vggt.models.vggt import VGGT
    params = inspect.signature(VGGT.__init__).parameters
    fastvggt = "merging" in params
    model = VGGT(merging=None, enable_point=point_head) if fastvggt else VGGT(enable_point=point_head, enable_track=False)
    sd = torch.load(weights or _download("facebook/VGGT-1B", "model.pt"), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    del sd
    if refocusing:
        load_refocusing(model, refocusing_weights)
    model = model.to(device).eval()
    if fastvggt:  # bfloat16 weights, as in the paper
        model = model.to(torch.bfloat16)
    return model


def load_pi3(device="cuda", weights="yyfz233/Pi3"):
    _add_path("ECOVGGT_PI3_ROOT", "Pi3")
    from pi3.models.pi3 import Pi3
    return Pi3.from_pretrained(weights, revision=REVISIONS.get(weights)).to(device).eval()


def load_da3(device="cuda", weights="depth-anything/DA3-LARGE"):
    root = _add_path("ECOVGGT_DA3_ROOT", "Depth-Anything-3")
    src = os.path.join(root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(weights, revision=REVISIONS.get(weights)).to(device).eval()


def load_vggt_omega(device="cuda", weights=None):
    """VGGT-Omega with the checkpoint of its authors for benchmarking (416 pixels). The weights on
    Hugging Face require accepting the license of VGGT-Omega first."""
    _add_path("ECOVGGT_VGGT_OMEGA_ROOT", "vggt-omega")
    from vggt_omega.models import VGGTOmega
    model = VGGTOmega()
    sd = torch.load(weights or _download("facebook/VGGT-Omega", "vggt_omega_1b_416_reproduce.pt"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    del sd
    return model.to(device).eval()


def load_backbone(name, device="cuda", **kwargs):
    """name: "vggt" (with Alignment Refocusing), "vggt-frozen" (without), "pi3", "da3" or "vggt-omega"."""
    if name == "vggt":
        return load_vggt(refocusing=True, device=device, **kwargs)
    if name == "vggt-frozen":
        return load_vggt(refocusing=False, device=device, **kwargs)
    if name == "pi3":
        return load_pi3(device=device, **kwargs)
    if name == "da3":
        return load_da3(device=device, **kwargs)
    if name == "vggt-omega":
        return load_vggt_omega(device=device, **kwargs)
    raise ValueError(f"unknown backbone {name!r}")
