# 🧠 Brain Tumor Segmentation using 3D U-Net and BraTS 2020

A PyTorch implementation of a **3D U-Net** for volumetric brain tumor segmentation, trained on the **BraTS 2020** dataset using four MRI modalities. The model segments three tumor sub-regions from 3D MRI volumes across 4 input channels.

---

## 📌 Overview

This project implements an end-to-end 3D medical image segmentation pipeline — from raw NIfTI MRI loading and preprocessing to model training, validation, and prediction visualization. The model is trained to segment brain tumors into three biologically meaningful sub-regions using multi-modal MRI data.

---

## 🗂️ Dataset

- **Dataset:** [BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- **Input Modalities (4 channels):** FLAIR, T1, T1CE, T2
- **Segmentation Classes (4):**

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

## 🏗️ Model Architecture — 3D U-Net

A custom 3D U-Net built from scratch with skip connections and transposed convolution upsampling.

```
UNET3D(
  in_channels  = 4   (FLAIR, T1, T1CE, T2)
  out_channels = 4   (background + 3 tumor regions)
  features     = [32, 64, 128, 256]
)
```

- **Encoder:** 4× DoubleConv3d blocks + MaxPool3d
- **Bottleneck:** DoubleConv3d(256 → 512)
- **Decoder:** ConvTranspose3d upsampling + skip connections + DoubleConv3d
- **Output:** Conv3d(32 → 4) with softmax

Each `DoubleConv3d` block: `Conv3d → BatchNorm3d → ReLU → Conv3d → BatchNorm3d → ReLU`

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 1 |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| Loss Function | DiceCELoss (MONAI) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Validation Metric | Mean Dice Score (MONAI DiceMetric) |
| Inference | Sliding Window (64×64×64 patches) |
| Validation Interval | Every 2 epochs |
| Device | GPU (CUDA) / CPU |

---

## 🔄 Checkpoint-Based Resumable Training

> **Training was done in sessions due to GPU time limits on Google Colab's free tier.**

To work around Colab's GPU session limits, a **best-model checkpoint strategy** was implemented:

- After every validation step, if the current mean Dice score improves over all previous epochs, the model weights are **automatically saved to Google Drive** (`best_brats_model.pth`)
- In the next available GPU session, the saved checkpoint is **reloaded and training resumes** from where it left off
- This ensures **no training progress is lost** between sessions

```python
# Save best model to Google Drive
if mean_metric > best_metric:
    best_metric = mean_metric
    torch.save(model.state_dict(), "/content/drive/MyDrive/best_brats_model.pth")

# Resume in next session
model.load_state_dict(torch.load("/content/drive/MyDrive/best_brats_model.pth"))
```

This approach simulates **continuous training across multiple interrupted sessions**, making it practical for anyone working under free-tier GPU constraints.

---

## 🛠️ Preprocessing Pipeline (MONAI Transforms)

- `LoadImaged` — Load NIfTI volumes
- `EnsureChannelFirstd` — Reorder to channel-first format
- `NormalizeIntensityd` — Per-channel, non-zero voxel normalization
- `Lambdad` — Relabel class 4 → 3 for contiguous label indexing
- `RandCropByPosNegLabeld` — Random 64×64×64 crops (pos:neg ratio = 3:1)
- `EnsureTyped` — Convert to PyTorch tensors

---

## 📦 Requirements

```bash
pip install torch torchvision monai torchmetrics mlxtend kagglehub matplotlib
```

Or in Google Colab:

```bash
!pip install -q torchmetrics mlxtend monai kagglehub
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SadeepM/Brain-Tumor-Segmentation-using-3D-UNet-and-BraTS2020.git
   ```

2. Open the notebook in [Google Colab](https://colab.research.google.com/):
   ```
   BraTS2020.ipynb
   ```

3. Mount Google Drive (for checkpoint saving):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Run all cells from top to bottom.
   - To **resume training** from a saved checkpoint, run the checkpoint loading cell before the training loop.

---

## 📊 What's Covered

- ✅ Downloading BraTS 2020 dataset via KaggleHub
- ✅ Loading and preprocessing 3D NIfTI MRI volumes with MONAI
- ✅ Multi-modal input stacking (FLAIR, T1, T1CE, T2)
- ✅ Label relabeling for contiguous class indexing
- ✅ Visualizing MRI slices and segmentation masks per modality
- ✅ Building 3D U-Net from scratch with skip connections
- ✅ Training with DiceCELoss and Adam optimizer
- ✅ Per-class and mean Dice Score validation
- ✅ Learning rate scheduling with ReduceLROnPlateau
- ✅ Sliding window inference for large 3D volumes
- ✅ Checkpoint saving to Google Drive after every best epoch
- ✅ Resuming training from saved checkpoint across GPU sessions
- ✅ Prediction visualization overlaid on MRI slices

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [MONAI](https://monai.io/) — Medical imaging transforms, losses, metrics, and inference
- [KaggleHub](https://github.com/Kaggle/kagglehub) — Dataset download
- [Matplotlib](https://matplotlib.org/)
- Google Colab (T4 GPU)
- Google Drive — Checkpoint storage and training resumption

---

## 📁 Project Structure

```
📦 repo
 ┗ 📓 BraTS2020.ipynb
 ┗ 📄 README.md
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
