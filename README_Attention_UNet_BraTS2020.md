# 🧠 Brain Tumor Segmentation using Attention U-Net and BraTS 2020

A PyTorch implementation of a **3D Attention U-Net** for volumetric brain tumor segmentation, trained on the **BraTS 2020** dataset using four MRI modalities. Attention gates are used to suppress irrelevant regions and focus on tumor-relevant features during decoding.

---

## 📌 Overview

This project extends the standard 3D U-Net with **Attention Gates** — a mechanism that selectively weighs skip connection features based on the decoder's gating signal. This allows the model to focus on relevant tumor regions and suppress background noise during upsampling, leading to sharper segmentation boundaries.

---

## 🗂️ Dataset

- **Dataset:** [BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- **Input Modalities (4 channels):** FLAIR, T1, T1CE, T2
- **Segmentation Classes:**

| Label | Region |
|-------|--------|
| 0 | Background |
| 1 | Necrotic / Non-enhancing Tumor Core (NCR/NET) |
| 2 | Peritumoral Edema (ED) |
| 3 | Enhancing Tumor (ET) *(relabeled from 4)* |

- **Data split:**
  - Training: ~72%
  - Validation: ~13%
  - Test: ~15%

> ⚠️ Subject `BraTS20_355` is skipped due to inconsistent file naming.

---

## 🏗️ Model Architecture — 3D Attention U-Net

Built from scratch with an encoder-decoder structure, **Attention Gates** on every skip connection, and trilinear upsampling in the decoder.

```
Attention_UNET3D(
  in_channels  = 4   (FLAIR, T1, T1CE, T2)
  out_channels = 4   (background + 3 tumor regions)
  features     = [32, 64, 128, 256]
)
```

**Encoder:** 4× DoubleConv3d blocks + MaxPool3d(2)

**Bottleneck:** DoubleConv3d(256 → 512)

**Decoder (per level):**
1. `Conv3d(feature×2 → feature, kernel=1)` — channel reduction
2. `F.interpolate(scale_factor=2, trilinear)` — upsampling
3. `AttentionGate(g, skip)` — attended skip connection
4. `Concat(attended_skip, x)` → `DoubleConv3d` — feature fusion

**Output:** `Conv3d(32 → 4, kernel=1)`

Each `DoubleConv3d` block:
```
Conv3d → BatchNorm3d → ReLU → Conv3d → BatchNorm3d → ReLU
```

### 🔍 Attention Gate

```python
class AttentionGate(nn.Module):
    # g_channels: gating signal (from decoder)
    # s_channels: skip connection (from encoder)
    # out_channels: intermediate channels (feature // 2)
    Wg: Conv3d(g_channels → out_channels, k=1) + BatchNorm3d
    Ws: Conv3d(s_channels → out_channels, k=1) + BatchNorm3d
    psi: Conv3d(out_channels → 1, k=1) + Sigmoid
    output: skip * psi(ReLU(Wg(g) + Ws(s)))
```

The gate computes a spatial attention map (values 0–1) from the gating signal and skip connection, then multiplies it element-wise with the skip features — suppressing irrelevant activations before concatenation.

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Max Epochs | 100 |
| Batch Size | 1 |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| Loss Function | DiceCELoss (MONAI) |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=5, min_lr=1e-6) |
| Validation Metric | Mean Dice Score (MONAI DiceMetric) |
| Inference | Sliding Window (64×64×64 patches, sw_batch_size=4) |
| Validation Interval | Every 2 epochs |
| Device | GPU (CUDA) / CPU |

---

## 📈 Results

Performance on the held-out test set:

| Region | Dice Score |
|--------|------------|
| Class 1 — Necrotic / Non-enhancing Tumor Core (NCR/NET) | 0.6054 |
| Class 2 — Peritumoral Edema (ED) | 0.7118 |
| Class 3 — Enhancing Tumor (ET) | 0.7407 |
| **Mean Test Dice** | **0.6860** |

---

## ⚠️ Limitations and Future Work

During training, the model began showing signs of **overfitting around epoch 46** — validation Dice plateaued or degraded while training loss continued to decrease.

**Planned improvements currently in progress:**

- **Data Augmentation** — Adding random flips, rotations, intensity shifts, and elastic deformations to improve generalization
- **Larger Patch Size (96×96×96)** — Increasing from 64³ to 96³ to provide the model with more spatial context per crop, which should improve segmentation of larger tumor regions

---

## 🔄 Checkpoint-Based Resumable Training

> **Training was done in multiple sessions due to GPU time limits on Google Colab's free tier.**

A **dual-checkpoint strategy** was implemented to handle interrupted sessions:

| Checkpoint | File | Purpose |
|------------|------|---------|
| `every_Att_UNETbrats20_checkpoint.pth` | Saved after **every epoch** | Allows resuming from the exact last epoch |
| `Att_UNETbrats20_checkpoint.pth` | Saved only when **validation Dice improves** | Stores the best model weights for testing |

**How it works:**

1. At session start, the notebook checks Google Drive for an existing checkpoint
2. If `every_Att_UNETbrats20_checkpoint.pth` exists, it restores model weights, optimizer state, scheduler state, and last epoch — then continues seamlessly
3. Falls back to the best-model checkpoint if the epoch-level one is absent
4. If neither exists, training starts from scratch

```python
if os.path.exists(every_checkpoint_path):
    checkpoint = torch.load(every_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_metric = checkpoint["best_metric"]
```

This approach simulates **continuous training across multiple interrupted sessions**, making it practical for anyone working under free-tier GPU constraints.

---

## 🛠️ Preprocessing Pipeline (MONAI Transforms)

| Transform | Purpose |
|-----------|---------|
| `LoadImaged` | Load NIfTI volumes from disk |
| `EnsureChannelFirstd` | Reorder dimensions to channel-first |
| `NormalizeIntensityd` | Per-channel normalization on non-zero voxels |
| `Lambdad` | Relabel class 4 → 3 for contiguous label indexing |
| `RandCropByPosNegLabeld` | Random 64×64×64 crops (pos:neg = 3:1, 4 samples per volume) |
| `EnsureTyped` | Convert to PyTorch tensors |

---

## 📦 Requirements

```bash
pip install torch torchvision monai kagglehub matplotlib scikit-learn
```

Or in Google Colab:

```bash
!pip install -q torchmetrics mlxtend monai kagglehub
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SadeepM/Brain-Tumor-Segmentation-using-Attention-UNet-and-BraTS2020.git
   ```

2. Open the notebook in [Google Colab](https://colab.research.google.com/):
   ```
   Attention_U_net_BraTS2020.ipynb
   ```

3. Mount Google Drive for checkpoint saving and loading:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Run all cells from top to bottom.
   - **First run:** Training starts from scratch and saves checkpoints to Drive
   - **Subsequent runs:** Checkpoint is auto-detected and training resumes from the last completed epoch

---

## 📊 What's Covered

- ✅ Downloading BraTS 2020 dataset via KaggleHub
- ✅ Loading and preprocessing 3D NIfTI MRI volumes with MONAI
- ✅ Multi-modal input stacking (FLAIR, T1, T1CE, T2)
- ✅ Label relabeling for contiguous class indexing (4 → 3)
- ✅ Visualizing MRI slices and segmentation masks per modality
- ✅ Overlaying predicted masks on MRI slices
- ✅ Building Attention Gate module from scratch
- ✅ Building 3D Attention U-Net from scratch with attended skip connections
- ✅ Training with DiceCELoss and Adam optimizer
- ✅ Per-class and mean Dice Score validation
- ✅ Learning rate scheduling with ReduceLROnPlateau
- ✅ Sliding window inference for full 3D volumes
- ✅ Dual checkpoint strategy — epoch-level and best-model checkpoints
- ✅ Seamless training resumption across Colab GPU sessions
- ✅ Prediction visualization against ground truth labels

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [MONAI](https://monai.io/) — Medical imaging transforms, losses, metrics, and inference
- [KaggleHub](https://github.com/Kaggle/kagglehub) — Dataset download
- [scikit-learn](https://scikit-learn.org/) — Train/val/test splitting
- [Matplotlib](https://matplotlib.org/)
- Google Colab (T4 GPU)
- Google Drive — Checkpoint storage across training sessions

---

## 📁 Project Structure

```
📦 repo
 ┗ 📓 Attention_U_net_BraTS2020.ipynb
 ┗ 📄 README.md
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
