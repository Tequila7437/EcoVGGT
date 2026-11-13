<div align="center">

<h1>EcoVGGT: Pinned-Key Inference for Efficient and Streamable<br>Feed-Forward 3D Reconstruction from Long Videos</h1>

Hongsen Cao&ensp;·&ensp;Shanxin Yuan&ensp;·&ensp;Mona Jaber&ensp;·&ensp;Ahmed Sayed

Queen Mary University of London

<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://huggingface.co/Tequila7437/EcoVGGT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Alignment%20Refocusing-f5a623" alt="Weights on Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/Code-VGGT%20License-2a78d6" alt="Code license"></a>
<a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/Weights-CC%20BY--NC%204.0-6f6e6a" alt="Weights license"></a>

<img src="assets/teaser_online.gif" width="100%" alt="Online reconstruction of a ScanNet video with EcoVGGT">

<i>The Online Stream of EcoVGGT (π³) on a 1,000-frame ScanNet video. Frames arrive on the left, and the reconstruction on the right grows as the stream emits them.</i>

</div>

## 📰 News

- **2026-11**: Code, Alignment Refocusing weights, and paper released.

<details>
<summary><b>Contents</b></summary>

[Overview](#-overview) · [Results](#-results) · [Installation](#-installation) · [Quick start](#-quick-start) · [Backbones](#-backbones) · [Online Stream](#-online-stream) · [Alignment Score](#-alignment-score) · [Evaluation](#-evaluation) · [Training](#-training-alignment-refocusing) · [Model weights](#-model-weights) · [Citation](#-citation) · [License](#-license)

</details>

## 🌟 Overview

Feed-forward multi-view transformers such as VGGT recover the cameras and geometry of a video in one pass, but the cost of their cross-frame self-attention grows quadratically with the number of frames, and their accuracy drops on long videos. EcoVGGT is a training-free inference framework in which every frame reads the same small set of keys, the Pinned Key Set of 31 frames spread evenly over the video. A pin pass stores the tokens of these frames in a Scene Cache, and all other frames read the cache in independent batches. These batches reproduce joint inference with the pinned keys exactly, up to floating-point rounding, so the cost per frame and the peak memory stay constant as the video grows. An Online Stream updates the pinned frames while a video is being recorded.

In VGGT, global blocks 10 to 15 align the frames. The Alignment Score reads their attention to flag likely failures without training or ground truth, and Alignment Refocusing fine-tunes only these six blocks (76M parameters, 3.6 GPU hours) to reduce failures. EcoVGGT runs on VGGT, π³, VGGT-Ω, and Depth Anything 3 (DA3), and we name each version after its backbone.

<p align="center"><img src="assets/pipeline.png" width="100%" alt="Overview of EcoVGGT"></p>

## 📊 Results

On ScanNet, EcoVGGT (π³) matches the success rate of π³ on 1,000-frame videos at 13.8 times its speed, where success means an absolute trajectory error (ATE) below 0.2 m. With Alignment Refocusing, EcoVGGT (VGGT) raises the success rate of VGGT from 70% to 82% on 500-frame videos at 8.5 times its speed. EcoVGGT reconstructs 4,000 frames on one GPU, and EcoVGGT (π³) streams 1,000 frames at 18.6 frames per second.

<div align="center">
<table>
<thead>
<tr><th rowspan="2" align="left">Method</th><th colspan="4">500 frames</th><th colspan="4">1,000 frames</th></tr>
<tr><th>Success (%)</th><th>ATE (m)</th><th>Time (s)</th><th>Memory (GB)</th><th>Success (%)</th><th>ATE (m)</th><th>Time (s)</th><th>Memory (GB)</th></tr>
</thead>
<tbody>
<tr><td>VGGT</td><td align="center">70</td><td align="center">0.111</td><td align="center">196</td><td align="center">23.1</td><td align="center">70</td><td align="center">0.130</td><td align="center">772</td><td align="center">44.2</td></tr>
<tr><td>EcoVGGT (VGGT)</td><td align="center">82</td><td align="center">0.090</td><td align="center">23.1</td><td align="center">10.0</td><td align="center">82</td><td align="center">0.093</td><td align="center">46.5</td><td align="center">10.0</td></tr>
<tr><td>EcoVGGT (VGGT), online</td><td align="center">78</td><td align="center">0.102</td><td align="center">45.8</td><td align="center">11.4</td><td align="center">78</td><td align="center">0.099</td><td align="center">64.6</td><td align="center">11.5</td></tr>
<tr><td>π³</td><td align="center">94</td><td align="center">0.054</td><td align="center">138</td><td align="center">24.4</td><td align="center">94</td><td align="center">0.058</td><td align="center">557</td><td align="center">43.4</td></tr>
<tr><td>EcoVGGT (π³)</td><td align="center">94</td><td align="center">0.052</td><td align="center">19.9</td><td align="center">9.9</td><td align="center">94</td><td align="center">0.060</td><td align="center">40.4</td><td align="center">9.9</td></tr>
<tr><td>EcoVGGT (π³), online</td><td align="center">90</td><td align="center">0.061</td><td align="center">37.0</td><td align="center">11.0</td><td align="center">92</td><td align="center">0.061</td><td align="center">53.8</td><td align="center">11.0</td></tr>
</tbody>
</table>
</div>

The table repeats the rows of the full models and of EcoVGGT from Table 1 of the paper, which compares them with 16 other methods on 50 ScanNet videos per length, all run on one A100 GPU with 80 GB. Time is the median per video, and memory is the largest peak memory of inference over the videos, without the input frames. EcoVGGT (VGGT) uses Alignment Refocusing.

<p align="center"><img src="assets/compare_scene0477.gif" width="100%" alt="Reconstructions of a long ScanNet video"></p>

The animation shows reconstructions of a 4,401-frame ScanNet video from 1,000 or 4,000 of its frames, viewed from above with the ceiling removed, with the ATE of each. InfiniteVGGT misplaces the walls, while EcoVGGT (π³) keeps every wall in place at 1,000 and 4,000 frames.

<p align="center"><img src="assets/tradeoff.png" width="100%" alt="Accuracy and cost on ScanNet"></p>

The plots show the success rate against speed at 1,000 frames, and the time per frame and the peak memory against the length of the video.

## 🔧 Installation

```bash
git clone https://github.com/Tequila7437/EcoVGGT.git
cd EcoVGGT
conda create -n ecovggt python=3.10 -y
conda activate ecovggt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
bash scripts/setup_third_party.sh
```

The setup script clones FastVGGT, whose VGGT implementation the paper uses, and π³ into `third_party/` at the commits of our experiments. For the DA3 backbone, run `bash scripts/setup_third_party.sh da3` and `pip install -e third_party/Depth-Anything-3`. For VGGT-Ω, run `bash scripts/setup_third_party.sh vggt-omega` and accept the license on its [Hugging Face page](https://huggingface.co/facebook/VGGT-Omega) before the first download. Training needs the losses of the official VGGT repository (`bash scripts/setup_third_party.sh vggt`), and evaluation needs `pip install -e ".[eval]"`. We tested the code with Python 3.10 and PyTorch 2.11 on CUDA 12.8. The VGGT backbones run in bfloat16 and need a GPU of the NVIDIA Ampere generation or newer.

## 🚀 Quick start

```python
import ecovggt
from ecovggt.io import load_images

images = load_images("path/to/frames")   # [S, 3, H, W] in [0, 1], 518 pixels wide
model = ecovggt.load_backbone("pi3")     # or "vggt", "vggt-frozen", "da3", "vggt-omega"
rec = ecovggt.reconstruct(model, images.cuda(), "pi3")

rec["w2c"]           # [S, 3, 4] world-to-camera matrices (OpenCV convention), world = camera of frame 0
rec["intrinsics"]    # [S, 3, 3]
rec["depth"]         # [S, H, W] depth maps, with confidence in rec["conf"]
rec["world_points"]  # [S, H, W, 3]
```

`load_images` reads a folder of images in natural name order or a video file. For VGGT-Ω, pass `backbone="vggt-omega"` to apply its own preprocessing at 416 pixels. The weights download on first use, and `load_backbone` accepts local files through `weights=` and, for `"vggt"`, `refocusing_weights=`. The command-line demo writes the cameras (`cameras.json`), the depth maps (`depth.npz`), and a colored point cloud (`points.ply`):

```bash
python demo.py --input path/to/frames --backbone pi3 --out outputs/scene
python demo.py --input video.mp4 --stride 5 --backbone vggt --score --out outputs/video
python demo.py --input path/to/frames --backbone pi3 --online --key_interval 32 --out outputs/stream
```

## 🧩 Backbones

| Name | Network | Offline | Online Stream | Alignment Score |
|---|---|:---:|:---:|:---:|
| `vggt` | VGGT-1B with Alignment Refocusing | yes | yes | yes |
| `vggt-frozen` | VGGT-1B with its released weights | yes | yes | yes |
| `pi3` | π³ | yes | yes | no |
| `da3` | DA3-LARGE | yes | no | no |
| `vggt-omega` | VGGT-Ω (checkpoint of its authors for benchmarking, 416 pixels) | yes | yes | no |

As in the paper, the VGGT backbones run the VGGT implementation of FastVGGT with bfloat16 weights, π³ and DA3 keep float32 weights under bfloat16 autocast, and VGGT-Ω follows its official forward pass, with the aggregator under bfloat16 autocast and both heads in float32. Every version pins 31 frames and attaches the other frames in batches of 64 (`num_keys` and `chunk` in `ecovggt.reconstruct`). The GPU memory of a pass depends on these two numbers and not on the length of the video.

## 📡 Online Stream

The Online Stream processes frames in the order of their arrival. Every `key_interval` frames, the newest frame becomes a candidate key, and each pass pins at most `max_keys` candidates spread evenly over the past. With a lag of `lag` passes, each pass emits the frames up to the candidate `lag` passes back. Each emitted frame then reads keys from the whole past and from up to `lag + 1` later candidates, and every frame is emitted within `(lag + 1) * key_interval` frames of its arrival.

```python
stream = ecovggt.Pi3OnlineStream(model, key_interval=32, max_keys=32)
for image in frames:            # [3, H, W] in [0, 1] on the GPU, one at a time
    emitted = stream.push(image)  # indices of the frames that this call completes
stream.finish()                 # emits the remaining frames
out = stream.results()
```

`ecovggt.VGGTOnlineStream(model, key_interval, lag=2, max_keys=32)` works in the same way on the VGGT backbones, and `ecovggt.vggt_omega_engine.online_reconstruct` runs the same stream over a recorded clip on VGGT-Ω. `demo.py --online` shows how to convert the outputs of every stream to cameras and points. The paper uses a key interval of 16 up to 500 frames and 32 for longer videos, at most 32 keys, a lag of 2 for VGGT, and no lag for π³.

## 🔎 Alignment Score

```python
model = ecovggt.load_backbone("vggt")
rec = ecovggt.reconstruct(model, images.cuda(), "vggt", score=True)
if rec["alignment_score"] < ecovggt.THRESHOLD_PINNED_KEYS:
    print("the reconstruction has likely failed")
```

The Alignment Score is the 90th percentile of the divergence of attention from uniform over 2,048 random queries and all heads of global block 13 in the pin pass. It is available before any attach pass and needs no training or ground truth, and a low score flags a likely failure. On 500-frame ScanNet videos, the paper sets the threshold to 9.09 with the Pinned Key Set and to 7.83 under full attention. On longer videos, the best threshold depends on the length. The score applies to the VGGT backbones and does not detect the failures of π³.

## 🧪 Evaluation

Download the `.sens` files of ScanNet after accepting its [terms of use](http://www.scan-net.org/), then export the 50 evaluation scenes:

```bash
python eval/prepare_scannet.py --sens_root /data/scannet/scans --scenes eval/scannet_50.txt \
    --out /data/scannet/process_scannet
```

Each command below evaluates one row of the main table of the paper:

```bash
python eval/eval_scannet.py --data_dir /data/scannet/process_scannet --sens_root /data/scannet/scans \
    --backbone vggt --num_frames 500 --out results/vggt_500
python eval/eval_scannet.py --data_dir /data/scannet/process_scannet --sens_root /data/scannet/scans \
    --backbone pi3 --num_frames 1000 --online --out results/pi3_online_1000
```

Each video keeps its first frame and takes the other frames at the largest integer stride that fills the frame budget. The script writes the metrics of each scene and a summary with the success rate and the medians over scenes of the ATE, the relative rotation and translation errors (RPE-r and RPE-t), the point accuracy and completeness, the absolute relative depth error (AbsRel), the time, and the peak memory. As in the paper, the point metrics use the world points that π³ and DA3 predict and the unprojected depth maps of VGGT and VGGT-Ω. Without `--sens_root`, the script computes the trajectory metrics only, which is what the paper reports at 2,000 and 4,000 frames together with time and memory. Our timings come from A100 GPUs with 80 GB. On other GPUs, the metrics can differ slightly from those of the paper, since bfloat16 kernels differ between GPU models.

## 🎯 Training Alignment Refocusing

```bash
bash scripts/setup_third_party.sh vggt
python eval/prepare_scannet.py --sens_root /data/scannet/scans --scenes training/scannet_train_195.txt \
    --out /data/scannet/process_scannet_train --train
python training/train_refocusing.py --data_dir /data/scannet/process_scannet_train \
    --scenes training/scannet_train_195.txt --weights /path/to/VGGT-1B/model.pt --out runs/refocusing
```

The 195 training scans are the first scans of ScanNet rooms that do not appear among the evaluation scenes. Each step pins 24 evenly spread frames of a whole scan or of a random contiguous half and draws 24 random other frames that read only these keys. The loss is the camera loss of VGGT on all frames plus its depth loss on six of them. The release weights come from 2,000 AdamW steps at a learning rate of 1e-5, which took 3.6 hours on one A100 GPU with a peak memory of 37 GB. Load a trained checkpoint with `ecovggt.load_backbone("vggt", refocusing_weights="runs/refocusing/alignment_refocusing_step2000.safetensors")`.

## 📦 Model weights

The Alignment Refocusing weights (`alignment_refocusing_vggt1b.safetensors`, 302 MB) replace global blocks 10 to 15 of VGGT-1B and hold 75.6M parameters. They are hosted on [Hugging Face](https://huggingface.co/Tequila7437/EcoVGGT), and `ecovggt.load_backbone("vggt")` downloads them together with VGGT-1B. The other backbones use the weights of their authors.

## 📝 Citation

```bibtex
@article{cao2026ecovggt,
  title   = {{EcoVGGT}: Pinned-Key Inference for Efficient and Streamable Feed-Forward {3D} Reconstruction from Long Videos},
  author  = {Cao, Hongsen and Yuan, Shanxin and Jaber, Mona and Sayed, Ahmed},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## 🙏 Acknowledgements

EcoVGGT builds on [VGGT](https://github.com/facebookresearch/vggt), [FastVGGT](https://github.com/mystorm16/FastVGGT), [π³](https://github.com/yyfz/Pi3), [VGGT-Ω](https://github.com/facebookresearch/vggt-omega), and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3). We thank their authors for releasing their code and weights. The evaluation uses [ScanNet](http://www.scan-net.org/), [evo](https://github.com/MichaelGrupp/evo), and [Open3D](https://www.open3d.org/).

## 📄 License

The code is a derivative of VGGT and is released under the [VGGT License](LICENSE). The parts adapted from π³, Depth Anything 3, and VGGT-Ω keep their licenses, as [NOTICE.md](NOTICE.md) lists, and the license of VGGT-Ω limits its engine to non-commercial research. The Alignment Refocusing weights are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial research, the license of the VGGT-1B weights from which they are fine-tuned.
