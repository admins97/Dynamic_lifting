# Unified-Lift (CVPR 2025)
## [[Paper](https://github.com/Runsong123/Unified-Lift/blob/main/material/Unified_Lift.pdf)]

> **Runsong Zhu**, Shi Qiu, Zhengzhe Liu, Ka-Hei Hui, Qianyi Wu, Pheng-Ann Heng, Chi-Wing Fu
> 

>**TL;DR**: Our paper presents a new and effective end-to-end lifting framework that achieves state-of-the-art (SOTA) performance for 3D Gaussian segmentation, without the need for pre-processing (e.g., video tracking) or post-processing (e.g., clustering).



### Comparisons with existing works

![image](https://github.com/Runsong123/Unified-Lift/blob/main/material/Teaser.png)

### 3D point-level visualization
![image](https://github.com/Runsong123/Unified-Lift/blob/main/material/3D_Segmentation.png)


### How to use the code. 
To use Unified-Lift, please refer to the [Usage](doc/Usage.md) guide. (currently under construction).


---

## This fork: two extensions + reproducible Docker setup

This fork adds two methodological extensions on top of the original object-level
codebook, plus a self-contained Docker environment for training and evaluation.

### 1. Adaptive codebook (revival / splitting)

The original codebook has a fixed size (256), so most entries stay unused on
scenes with few objects while busy entries may have to cover more than one
instance. We track an EMA of each code's usage (the soft-assignment mass from
confident pixels) and periodically **recycle dead codes by splitting an
over-loaded "donor" code into two perturbed children** (VQ-VAE-style revival,
with the corresponding Adam moments reset). The effective number of objects is
therefore discovered from the data instead of being fixed in advance.

Implemented in `update_code_usage()` / `revive_codebook()` in
[`train_unified_lift.py`](train_unified_lift.py). Controlled by `--revive_interval`
(default `1000`, `0` disables); revival runs in the window
`(max(reg3d_from, 2000), 0.8 * iterations)`.

### 2. 3D instance compactness loss

Pure 2D supervision cannot tell apart Gaussians that share a code but sit far
apart in 3D (floaters / split objects). We soft-assign every (sub-sampled)
Gaussian to the codebook via its per-Gaussian feature and **penalise the
assignment-weighted spatial variance of each code**, normalised by the scene
scale (xyz detached so the loss shapes the features/codebook, not the geometry).

Implemented in `compactness_loss_3d()` in
[`train_unified_lift.py`](train_unified_lift.py). Controlled by `--weight_3d`
(default `0.1`) and `--reg3d_from` (default `3000`).

### Results (LERF-mask, iteration 30000)

Reproduced in this fork (Docker, RTX A6000). Rendering uses the model's own
RGB to drive Grounded-SAM for the text-query evaluation.

| Scene      | mIoU  | Boundary IoU |
|------------|-------|--------------|
| figurines  | 85.03 | 83.41 |
| ramen      | 78.27 | 69.77 |
| teatime    | 78.46 | 74.41 |
| **average**| **80.59** | **75.86** |

### Docker usage

```bash
# 1. Build the training image (CUDA 11.3 / PyTorch 1.12.1, builds the CUDA submodules)
docker build -t unified-lift:cu113 .

# 2. Train. Datasets (figurines/ramen/teatime, LERF-mask format) live one level
#    above the repo and are mounted at /data.
./docker_train.sh figurines 0          # <scene> <gpu_id> [extra train args]

# 3. Build the evaluation image (adds Grounded-SAM = GroundingDINO + SAM)
docker build -f Dockerfile.eval -t unified-lift:eval .
#    Place SAM ViT-H weights at Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth
#    (GroundingDINO weights are fetched from Hugging Face and cached in ./.hfcache).

# 4. Render text-query masks for all scenes and compute mIoU / bIoU
./docker_eval.sh 0                     # <gpu_id>; results -> results/Unifed_Lift.csv
```


# Citation
If you find this work useful in your research, please cite our paper:
```
@article{zhu2025rethinking,
  title={Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting},
  author={Zhu, Runsong and Qiu, Shi and Liu, Zhengzhe and Hui, Ka-Hei and Wu, Qianyi and Heng, Pheng-Ann and Fu, Chi-Wing},
  journal={arXiv preprint arXiv:2503.14029},
  year={2025}
}

```


## Thanks
This code is based on [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping), [OmniSeg3D-GS](https://github.com/OceanYing/OmniSeg3D-GS), [Panoptic-Lifting](https://github.com/nihalsid/panoptic-lifting). We thank the authors for releasing their code. 




