# Notices

EcoVGGT builds on the projects below. Their licenses apply to the parts that this table names.

| Project | License | Use in this repository |
|---|---|---|
| [VGGT](https://github.com/facebookresearch/vggt) | VGGT License ([LICENSE](LICENSE)) | `ecovggt/vggt_engine.py` and `ecovggt/alignment.py` run the modules of VGGT with code adapted from its aggregator, and `training/train_refocusing.py` uses its training losses. The Alignment Refocusing weights are fine-tuned from VGGT-1B. |
| [FastVGGT](https://github.com/mystorm16/FastVGGT) | VGGT License | Provides the VGGT implementation that the paper uses. `eval/` follows its ScanNet protocol, and `eval/scannet_50.txt` is its list of scenes. |
| [π³](https://github.com/yyfz/Pi3) | BSD 3-Clause ([licenses/LICENSE-Pi3-BSD-3-Clause.txt](licenses/LICENSE-Pi3-BSD-3-Clause.txt)) | `ecovggt/pi3_engine.py` adapts its decoder loop and head calls. |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Apache License 2.0 ([licenses/LICENSE-Depth-Anything-3-Apache-2.0.txt](licenses/LICENSE-Depth-Anything-3-Apache-2.0.txt)) | `ecovggt/da3_engine.py` adapts its layer loop and uses its camera utilities. |
| [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) | FAIR Noncommercial Research License ([licenses/LICENSE-VGGT-Omega-FAIR-Noncommercial.txt](licenses/LICENSE-VGGT-Omega-FAIR-Noncommercial.txt)) | `ecovggt/vggt_omega_engine.py` adapts its layer loop and is distributed under this license, which permits non-commercial research only. |

The code of this repository is a derivative of VGGT and is distributed under the VGGT License, as that license requires. The adapted parts of π³, Depth Anything 3, and VGGT-Omega keep their own licenses.

This repository contains no backbone weights. `scripts/setup_third_party.sh` clones the backbone code, and the weights download from the pages of their authors under their own licenses: [VGGT-1B](https://huggingface.co/facebook/VGGT-1B), [π³](https://huggingface.co/yyfz233/Pi3), [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE), and [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega), whose page asks users to accept its license before downloading. The Alignment Refocusing weights are released under CC BY-NC 4.0, the license of VGGT-1B, and were trained on ScanNet, whose terms of use also limit them to non-commercial research.
