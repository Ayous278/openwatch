# OpenWatch: A Multimodal Benchmark for Hand Gesture Recognition on Smartwatches

Code and benchmark artifacts for **"OpenWatch: A Multimodal Benchmark for Hand Gesture Recognition on Smartwatches"** by Pietro Bonazzi, Youssef Ahmed, Daniel Eckert, Michele Magno, Dengxin Dai (ETH Zürich, Huawei Research Zürich).

OpenWatch is the first open-access multi-modal benchmark for wrist-based gesture recognition. It contains over 10 hours of synchronized 6-axis IMU and PPG data from 50 participants and 78 sessions, covering 59 labeled gesture classes including a negative/background class.

This repository accompanies the paper and provides:

- The full training, evaluation, and ablation code (Jupyter notebooks).
- All small per-run artifacts produced during benchmarking: configs, metrics, training/validation curves, confusion matrices, classification reports, and history logs.
- Pointers to the **dataset** (Hugging Face) and **trained model weights** (Hugging Face).

## Repository layout

```
.
├── code/                     # Jupyter notebooks
│   ├── Augmentations.ipynb   # Data preprocessing + augmentation pipeline
│   ├── Benchmark.ipynb       # Main benchmark: trains and evaluates all model families
│   └── Normwear_ppg.ipynb    # NormWear + LoRA experiments and PPG-specific studies
├── Benchmark/                # Per-model artifacts (configs, metrics, plots, reports)
│   ├── apple/                # Apple-style filterbank CNN  (aug / no_aug)
│   ├── inceptiontime/        # InceptionTime baseline      (aug / no_aug)
│   ├── hydra/                # Hydra baseline              (aug / no_aug)
│   ├── transformer/          # Mix-Token (proposed)        (aug / no_aug)
│   ├── normwear_base/        # NormWear linear-probe       (IMU / IMU+PPG × aug / no_aug)
│   ├── normwear_lora/        # NormWear + LoRA (proposed)  (IMU / IMU+PPG × aug / no_aug)
│   ├── ablations_mask/       # Token-masking ablations for Mix-Token
│   └── HAR_PPG/              # UCI-HAR transfer / linear-probe results
├── requirements.txt
├── .gitignore
└── README.md
```

Trained weights (`*.pt`, `*.ckpt`) and large feature caches (`*.npy` over 1 MB) are **not in this repo**. See [Model weights](#model-weights).

## Dataset

The OpenWatch dataset is available on the Hugging Face Hub:

- https://huggingface.co/datasets/pietrobonazzi/openwatch

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="pietrobonazzi/openwatch",
    repo_type="dataset",
    local_dir="data/openwatch",
)
```

## Model weights

All trained checkpoints and embedding caches are hosted on the Hugging Face Hub at:

- `https://huggingface.co/Ayous278/openwatch-models`  *(update this URL if you host the weights under a different namespace)*

Weights are organized to mirror the `Benchmark/` directory in this repo, so you can drop them in place after download:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Ayous278/openwatch-models",
    repo_type="model",
    local_dir="Benchmark",          # merges into the Benchmark/ folder
    local_dir_use_symlinks=False,
)
PY
```

After this step, every run folder under `Benchmark/<family>/<run>/` will contain its `*.pt` / `*.ckpt` checkpoint and any `*_embeddings.npy` caches alongside the metrics and configs that already ship with the repo.

## Installation

The notebooks were developed and tested with Python 3.10 and CUDA-enabled PyTorch. We recommend a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The NormWear foundation model is imported as `from NormWear.main_model import NormWearModel`. Install it from its official source as referenced in the paper before running `Benchmark.ipynb` or `Normwear_ppg.ipynb`.

## Reproducing the paper

The notebooks read a few paths from environment variables. Set them once before opening Jupyter:

```bash
export OPENWATCH_ROOT="/abs/path/to/data/openwatch"   # local clone of the HF dataset
export UCI_ROOT="/abs/path/to/UCI_HAR"                # optional, only for HAR_PPG transfer
```

Then run, in order:

1. `code/Augmentations.ipynb` – preprocesses the raw OpenWatch recordings into the windowed train/val/test tensors used downstream.
2. `code/Benchmark.ipynb` – trains and evaluates Apple-CNN, InceptionTime, Hydra, Mix-Token (transformer), and NormWear (linear probe + LoRA) across the augmentation conditions reported in Table 2 of the paper.
3. `code/Normwear_ppg.ipynb` – runs the IMU+PPG variants of NormWear and the UCI-HAR transfer experiments.

Optional logging to Weights & Biases is enabled inside the notebooks; run `wandb login` first or comment out the WandB callbacks if you don't want logging.

## Citation

```bibtex
@inproceedings{bonazzi2026openwatch,
  title     = {OpenWatch: A Multimodal Benchmark for Hand Gesture Recognition on Smartwatches},
  author    = {Bonazzi, Pietro and Ahmed, Youssef and Eckert, Daniel and Magno, Michele and Dai, Dengxin},
  booktitle = {},
  year      = {2026}
}
```
