# AnatCoder

[![arXiv](https://img.shields.io/badge/arXiv-coming--soon-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

AnatCoder is an anatomy-conditioned neural implicit representation framework for sparse-view CT reconstruction with emergent segmentation.

## Overview

AnatCoder injects anatomical priors (Atlas probability maps) into an INR backbone so reconstruction and segmentation are learned jointly. The framework also supports anatomy-decomposed volume rendering (ADVR) for projection-domain regularization and provides a unified benchmark harness for sparse-view CT methods.

![Architecture](assets/architecture.png)

## Installation

```bash
git clone https://github.com/xxx/AnatCoder.git
cd AnatCoder
bash setup_env.sh
```

## Data Preparation

1. Download raw CT + segmentation datasets:

```bash
bash scripts/download_data.sh
```

2. Preprocess all cases:

```bash
python scripts/preprocess_all.py --input_dir data/raw --output_dir data/processed --crop_size 128
```

3. Generate sparse projections:

```bash
python scripts/generate_projections.py   --data_dir data/processed   --output_dir data/projections   --n_views 10 20 30 50   --geo_config configs/data/default.yaml
```

## Quick Start

```bash
# Train vanilla baseline
python train.py method=vanilla_inr data.n_views=50

# Train AnatCoder
python train.py method=anatcoder data.n_views=50

# Evaluate
python scripts/evaluate.py --ckpt outputs/anatcoder_50views/
```

## Supported Methods

| Method | Type | Open-source status |
| --- | --- | --- |
| AnatCoder | Ours | This repository |
| Vanilla-INR | Internal baseline | This repository |
| NAF (MICCAI'22) | External baseline | Public |
| Spener (AAAI'25) | External baseline | Public |
| SAX-NeRF (CVPR'24) | External baseline | Public |
| NAB (ICLR'26) | External baseline | Re-implementation planned |
| TP-INR (MICCAI'25) | External baseline | Pending / backup implementation |
| FDK / SART | Classical baseline | TIGRE built-in |

## Results

| Method | Views | PSNR | SSIM | MAE |
| --- | --- | --- | --- | --- |
| AnatCoder | 50 | TBD | TBD | TBD |
| Vanilla-INR | 50 | TBD | TBD | TBD |

## Citation

```bibtex
@article{anatcoder2027,
  title   = {AnatCoder: Anatomy-Conditioned Neural Implicit Encoding for Sparse-View CT Reconstruction with Emergent Segmentation},
  author  = {Zhang, Jian and others},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2027}
}
```

## Acknowledgments

This project builds on open-source ecosystems including PyTorch, Lightning, Hydra, TIGRE, tiny-cuda-nn, and the medical imaging community datasets TotalSegmentator and AMOS.
