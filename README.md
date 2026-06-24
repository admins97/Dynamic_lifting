# SKKU SELF 2026-1 (Dynamic Lifting)

Dynamic Lifting is an end-to-end framework for object-aware 3D Gaussian segmentation, built on **Unified-Lift** (CVPR 2025 — [paper](https://github.com/Runsong123/Unified-Lift/blob/main/material/Unified_Lift.pdf), [original repo](https://github.com/Runsong123/Unified-Lift)). It lifts 2D masks into a 3D Gaussian Splatting scene with no pre-processing (e.g., video tracking) or post-processing (e.g., clustering).

This repository adds two extensions to the object-level codebook — an adaptive codebook with usage-based revival/splitting, and a 3D instance compactness loss — together with a reproducible Docker setup for training and Grounded-SAM evaluation.

The code lives at the repository root; commands below are run from there.

## Setup

Datasets use the LERF-mask layout (COLMAP `sparse/`, `images_train/`, `object_mask/`, `test_mask/`) and live one level above the repository, so they mount at `/data` inside the container. Replace the example paths with your own.

The training and evaluation environments are provided as Docker images (see [Docker](#docker)); building them is the recommended setup. To run on a bare machine instead, the environment is:

```bash
conda create -n unifed_lift python=3.8 -y
conda activate unifed_lift
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1 tqdm scipy wandb opencv-python scikit-learn lpips
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```

## Extensions

### Adaptive codebook (revival / splitting)

The original codebook has a fixed size (256), so most entries stay unused on scenes with few objects while busy entries may have to cover more than one instance. We track an EMA of each code's usage (the soft-assignment mass from confident pixels) and periodically recycle dead codes by splitting an over-loaded "donor" code into two perturbed children (VQ-VAE-style revival, with the corresponding Adam moments reset). The effective number of objects is therefore discovered from the data instead of being fixed in advance.

Implemented in `update_code_usage()` / `revive_codebook()` in `train_unified_lift.py`. Controlled by `--revive_interval` (default `1000`, `0` disables); revival runs in the window `(max(reg3d_from, 2000), 0.8 * iterations)`.

### 3D instance compactness loss

Pure 2D supervision cannot tell apart Gaussians that share a code but sit far apart in 3D (floaters / split objects). We soft-assign every (sub-sampled) Gaussian to the codebook via its per-Gaussian feature and penalise the assignment-weighted spatial variance of each code, normalised by the scene scale (xyz detached so the loss shapes the features/codebook, not the geometry).

Implemented in `compactness_loss_3d()` in `train_unified_lift.py`. Controlled by `--weight_3d` (default `0.1`) and `--reg3d_from` (default `3000`).

## Train

Train a single scene in the training image (datasets mount at `/data`, outputs are written under `output/`):

```bash
./docker_train.sh figurines 0          # <scene> <gpu_id> [extra train args]
```

Train all three scenes:

```bash
for s in figurines ramen teatime; do ./docker_train.sh "$s" 0; done
```

Useful runtime overrides (passed straight through to the trainer):

```bash
./docker_train.sh figurines 0 --iterations 30000 --weight_3d 0.1 --reg3d_from 3000 --revive_interval 1000
```

Run the trainer directly, without the Docker helper:

```bash
CUDA_VISIBLE_DEVICES=0 python train_unified_lift.py \
  -s /data/figurines -m output/unifed_lift/figurines \
  --config_file config/gaussian_dataset/train.json --train_split --weight_loss 1e-0
```

## Inference and Evaluation

Render LERF text-query masks for all scenes and compute mIoU / bIoU:

```bash
./docker_eval.sh 0                     # <gpu_id>; results -> results/Unifed_Lift.csv
```

This renders each scene with `render_lerf_mask_unified_lift.py` (Grounded-SAM text query, driven by the model's own rendered RGB) and then scores predictions against `test_mask/` with `script/eval_lerf_mask_unified_lift.py`, writing per-scene mIoU / bIoU to `results/Unifed_Lift.csv`.

## Docker

Two images: training (`unified-lift:cu113`, CUDA 11.3 / PyTorch 1.12.1, builds the CUDA submodules) and evaluation with Grounded-SAM (`unified-lift:eval`, adds GroundingDINO + SAM).

Build:

```bash
docker build -t unified-lift:cu113 .
docker build -f Dockerfile.eval -t unified-lift:eval .
```

The eval image expects SAM ViT-H weights at `Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth`; GroundingDINO weights are fetched from Hugging Face and cached in `./.hfcache`.

Helper launchers:

```text
docker_train.sh <scene> <gpu_id>   train one scene in the training image
docker_eval.sh  <gpu_id>           render + evaluate all scenes in the eval image
```

Both mount the repository at `/workspace` and the dataset parent directory at `/data`. Edit the mount paths in the scripts before using them on your machine.

## Code Structure

```text
train_unified_lift.py              Training entry point (codebook + the two extensions)
render_lerf_mask_unified_lift.py   LERF text-query rendering (Grounded-SAM)
render.py                          Standard rendering / feature visualization
train.sh / infer.sh                Upstream multi-scene train / eval entry scripts
docker_train.sh / docker_eval.sh   Docker launchers
Dockerfile / Dockerfile.eval       Training / evaluation images
config/                            Dataset and training configs (JSON)
arguments/                         Command-line, model, and optimization parameters
scene/                             Scene + Gaussian model, COLMAP / dataset readers
gaussian_renderer/                 Differentiable Gaussian rasterization wrapper
submodules/                        diff-gaussian-rasterization, simple-knn (CUDA ext)
ext/                               Grounded-SAM glue (GroundingDINO + SAM)
utils/                             Loss, image, and general utilities
script/                            Evaluation script (mIoU / bIoU)
Tracking-Anything-with-DEVA/       SAM checkpoint location (saves/) + DEVA assets
lama/                              Inpainting assets (upstream)
test_newfuncs.py                   Unit tests for the codebook / compactness additions
doc/                               Upstream usage notes
```

## Notes

- Static assets (datasets, checkpoints, model weights, generated outputs, W&B logs) are intentionally not committed; see `.gitignore`.
- Training outputs are written under the run directory passed with `-m` (e.g., `output/unifed_lift/<scene>/`); rendered masks and metrics land under that model path and `results/`.
- Datasets are not bundled — place `figurines` / `ramen` / `teatime` (LERF-mask format) one level above the repository.

## Citation

This work builds on Unified-Lift; if you use it, please cite the original paper:

```
@article{zhu2025rethinking,
  title={Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting},
  author={Zhu, Runsong and Qiu, Shi and Liu, Zhengzhe and Hui, Ka-Hei and Wu, Qianyi and Heng, Pheng-Ann and Fu, Chi-Wing},
  journal={arXiv preprint arXiv:2503.14029},
  year={2025}
}
```

## Thanks

This code is based on [Unified-Lift](https://github.com/Runsong123/Unified-Lift), [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping), [OmniSeg3D-GS](https://github.com/OceanYing/OmniSeg3D-GS), and [Panoptic-Lifting](https://github.com/nihalsid/panoptic-lifting). We thank the authors for releasing their code.
