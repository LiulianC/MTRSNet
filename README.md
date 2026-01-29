## MTRSNet for Endoscopic images and videos

---

### Demo
---
<p align="center">
  <img src="assert/demo.gif" alt="Video Demo" width="600"/>
</p>

### Introduction
---
<p align="center">
  <img src="assert/arch.png" alt="Network Architecture" width="600"/>
</p>

Given an original image (Input image), we first extract multi-scale high-frequency reflection features using
a Laplacian Pyramid. Then, we use a Residual-SE Layer to extract and condense the features of the original
image and the reflection prior. Finally, the two feature maps are fused and concatenated into a tensor with 64
channels. The features are then processed by a dual-execution encoder composed of two cooperative branches:
a high-efficiency Mamba pathway that models long-)ange dependencies through a four-directional scan with
input-conditioned discretization, and a Swin-Transformer pathway that preserves localized high-frequency details. A self-scheduling policy balances the contributions of the two branches over training, and their outputs
are fused by a pixel-wise attention aggregator that assigns larger weights to fine-texture pixels prone to specular
washout. The fused multi-scale representation enters a multilevel cross-feedback convergent stage (MCCS),
where features propagate across sub-networks and depths via up/down-sampling feedback links, driving a consensus of global context and local structures and concentrating the richest semantics at the deepest node. Finally,
a dual-path affine residual coupled decoder (DP-ARCD) progressively merges hierarchical features and jointly
estimates the residuals for the transmission and reflection components; the predicted residuals are scaled by
learnable factors and added back to the input to yield the reflection-free output T alongside R. Throughout the
pipeline, the four-directional Mamba blocks are implemented with kernel-fused CUDA/Triton operators to reduce off-chip I/O, ensuring that the model remains both computationally efficient and effective at disentangling
T from R in high-resolution endoscopic imagery.

### Environment
---
- **Ubuntu**: 22.04
- **Python**: 3.10
- **PyTorch**: 2.6.0
- **CUDA**: 12.4

### Installation
---
Install required packages:
```
pip install opencv-python pillow matplotlib scikit-image scipy pyyaml tqdm tabulate ipdb tensorflow timm triton accimage
```

[Install Mamba-ssm and causal-conv1d](https://github.com/state-spaces/mamba/issues/808#issuecomment-3719102259)

### Dataset and Preparation
---

**Available Datasets:**
- [HyperKavsir](https://github.com/researchmm/STTN) - Endoscopic image dataset
- [MouseData](https://drive.google.com/file/d/1eOiaGrRYSL9kGgwFj9ZiFTTTD60mahUQ/view?usp=sharing) - Mouse data dataset

**Dataset Descriptions:**
- **HyperKavsir**: Uses the `HyperKDataset` interface. Requires inter-frame restoration to obtain corresponding input-label pairs.
- **MouseData**: Uses the `DSRTestDataset` interface. Can be used directly.

### Training Hyperparameters
---

**1. Dataset Configuration Parameters**

| Parameter Name | Type | Default Value | Description |
|--------|------|--------|------|
| `data_root` | str | `'./data'` | Data root directory path |
| `batch_size_train` | int | 1 | Training batch size |
| `batch_size_test` | int | 4 | Testing batch size |
| `shuffle` | bool | True | Whether to shuffle data |
| `num_workers` | int | 0 | Number of data loading threads |
| `sampler_size1-5` | int | Depends on training | Sampling sizes for each data source |
| `test_size` | list | [200, 0, 0, 0, 200, 200] | Sample counts for each test set |

**2. Model Training Configuration**

| Parameter Name | Type | Default Value | Description |
|--------|------|--------|------|
| `epoch` | int | 121 | Total training epochs |
| `base_lr` | float | 1e-4 | Base learning rate |
| `scheduler_type` | str | 'plateau' | Learning rate scheduler type (plateau/cosine) |
| `es_patience` | int | 20 | Early stopping patience (epochs without improvement) |
| `es_delta` | float | 1e-4 | Early stopping minimum improvement threshold |
| `es_verbose` | bool | True | Whether early stopping prints information |

**3. Path and File Configuration**

| Parameter Name | Type | Default Value | Description |
|--------|------|--------|------|
| `model_dir` | str | `'./model_fit'` | Model save directory |
| `save_dir` | str | `'./results'` | Results save directory |
| `model_path` | str | `'./model_118.pth'` | Checkpoint path to load |
| `reset_best` | bool | False | Whether to reset best model record |

**4. Debug and Monitoring Configuration**

| Parameter Name | Type | Default Value | Description |
|--------|------|--------|------|
| `always_print` | int | 0 | Always print information |
| `debug_monitor_layer_stats` | int | 0 | Monitor layer statistics |
| `debug_monitor_layer_grad` | int | 0 | Monitor layer gradients |
| `display_id` | int | -1 | Display ID (for visualization) |
| `host` | str | '127.0.0.1' | Host address |
| `port` | int | 57117 | Port number |
| `throttle_ms` | int | 0 | Sleep milliseconds after each optimizer step |

**5. Feature Switch Configuration**

| Parameter Name | Type | Default Value | Description |
|--------|------|--------|------|
| `training` | bool | False | Whether training mode is enabled |
| `color_enhance` | bool | False | Whether color enhancement is enabled |
| `AdditionSkip_en` | bool | True | Whether additional skip connections are enabled |

**6. Learning Rate Mapping Configuration (Decoder-Specific)**

| Module Name | Learning Rate | Description |
|--------|--------|------|
| `token_decoder3` | 9.9e-05 | Final decoder layer learning rate |
| `token_decoder2` | 9.9e-05 | Third decoder layer learning rate |
| `token_decoder1` | 9.9e-05 | Second decoder layer learning rate |
| `token_decoder0` | 9.9e-05 | First decoder layer learning rate |



### Training
---

Run `train.py` to start model training. Before running, execute the following steps:

**Step 1**: Modify training parameters in `MTRR_option.py`

**Step 2**: In `train.py`, fill in the paths and index file paths for MouseData and HyperKavsir, then run. Other datasets are not involved in training by default.

After running `train.py`, the following directories and files will be automatically generated:

```
./indexcsv/                       # CSV log file directory
  ├── {timestamp}_train_loss.csv  # Training loss log (per epoch)
  └── {timestamp}_index.csv       # Validation metrics (PSNR, SSIM, LMSE, NCC)

./model_fit/                      # Model checkpoint directory
  ├── model_latest.pth            # Latest model
  └── model_{epoch}.pth           # Models for each epoch

./img_results/                    # Visualization results directory
  ├── output_train_{timestamp}/   # Generated images during training
  └── output_test_{timestamp}/    # Generated images during testing
```

**Content displayed during each epoch:**
- 📊 Real-time progress bar for current epoch and batch
- 💔 Real-time loss values: loss, mseloss, vggloss, ssimloss, loss_spr
- 📈 Current learning rate
- ✅ Validation metrics: PSNR, SSIM, LMSE, NCC
- ⛔ Early stopping alerts
- 💾 Model save status


### Inference
---

Run `inference.py` for model inference. Before running, you need to prepare:

**Preparation 1**: Model checkpoint file (.pth)
- Specify via the `--ckpt` parameter or configure as `model_path` in `MTRR_option.py`
- Used to load the trained MTRSNet model weights

**Preparation 2**: Dataset paths (hardcoded in the code)
- Tissue real-time data: `/home/hostname/hostname-MTRRVideo/data/tissue_real` and its index files
- Training set index: `train1.txt` (800 samples)
- Test set index: `eval1.txt` (200 samples)

**Example inference command:**
```bash
python inference.py --ckpt ./model_fit/model_latest.pth --outdir ./infer_outputs
```

After running `inference.py`, the following will be generated in `./infer_outputs/output_infer_{timestamp}/` directory:

**1. Transmission Layer Prediction Images**
- Filename: `{num:04d}-grid_fakeT.png`
- Description: Predicted transmission image (enhanced with histogram matching)
- Layout: Grid arrangement with 4 images per row

**2. Original Input Images** (Optional)
- Filename: `{num:04d}-grid_input.png`
- Description: Input original mixed image

**3. Reflection Layer Prediction Images** (Only when `--save-reflection` is specified)
- Filename: `{num:04d}-grid_fakeR.png`
- Description: Predicted reflection component

### Visualization
---

Visualization results as described above are automatically generated and saved during the training and inference process.

### Project Structure
---

**Complete MTRSNetv2 Project Directory Description**

```
Project Root Directory (./)
├── [Training and Inference Scripts]
│   ├── train.py                    - Main training script for MTRSNet model
│   ├── inference.py                - Inference script for predictions on dataset
│   ├── debug_train.py              - Debug training script
│   └── classifier.py               - Classifier-related code
│
├── [Model Architecture Files]
│   ├── MTRSNet.py                  - MTRSNet main architecture and engine class
│   ├── MTRR_RD_modules.py          - MTRR R-D module implementation
│   ├── MTRR_token_modules.py       - MTRR token module implementation
│   ├── vmamba.py                   - VMamba model implementation
│   └── reverse_function.py         - Reverse function implementation
│
├── [Configuration and Options]
│   ├── MTRR_option.py              - Training option configuration (learning rate, batch_size, etc.)
│   └── set_seed.py                 - Random seed setting
│
├── [Loss Functions]
|   ├── customloss.py               - Custom loss function
│   └── psdLoss/                    - Proprietary loss function package
│       ├── losses.py               - Main loss function definition
│       ├── focal_loss.py           - Focal loss
│       ├── lovasz_losses.py        - Lovász loss
│       ├── ssim.py                 - SSIM loss
│       ├── vgg.py                  - VGG feature extraction loss
│       ├── spec_loss_pack.py       - Specular reflection loss package
│       └── CX/                     - Context loss module
│           ├── CX_distance.py
│           ├── CX_helper.py
│           └── enums.py
│
├── [Dataset Processing]
│   └── dataset/                    - Dataset loading and preprocessing
│       ├── new_dataset1.py         - Dataset class definition
│       ├── quality_index.py        - Image quality assessment metrics
│       ├── transforms.py           - Data augmentation transformations
│       ├── torchdata.py            - PyTorch data loader
│       ├── image_folder.py         - Image folder reading
│       ├── hook.py                 - Data loading hooks
│       └── util.py                 - Data processing utility functions
│
├── [Utility Functions]
│   ├── util/                       - General utility module
│   │   ├── color_enhance.py        - Color enhancement and histogram matching
│   │   ├── eval_util.py            - Evaluation utility functions
│   │   ├── csv.py                  - CSV file writing tool
│   │   ├── cupcut.py               - Image cropping tool
│   │   └── video_grip.py           - Video processing tool
│   ├── core/                       - Core utilities
│   │   └── utils.py                - General utility functions
│   ├── rcmap_mask_fusion.py        - Reflection map mask fusion
│   ├── video_func.py               - Video processing functions
│   └── early_stop.py               - Early stopping mechanism
│
├── [Performance and Analysis]
│   ├── calc_flops_test.py          - FLOPS calculation test
│   ├── calc_throughout.py          - Throughput calculation
│   └── flops.md                    - FLOPS analysis document
│
├── [Batch Processing]
│   └── batch_zip_infer.py          - Batch inference script
│
├── [Documentation]
│   ├── README_cn.md                - Chinese documentation
│   └── README_en.md                - English documentation
│
├── [Resources Directory]
│   └── assert/                     - Resource files directory
│
└── [Version Control]
    ├── .git/                       - Git version control directory
    └── .gitignore                  - Git ignore configuration
```

### File Function Quick Reference
---

**Core Training Pipeline:**
```
train.py → MTRR_option.py → MTRSNet.py → psdLoss/
```

**Inference Pipeline:**
```
inference.py → MTRSNet.py → dataset/new_dataset1.py
```

**Data Processing Pipeline:**
```
dataset/new_dataset1.py → dataset/transforms.py → dataset/quality_index.py
```

**Loss Function System:**
```
customloss.py → psdLoss/losses.py + psdLoss/ssim.py + psdLoss/vgg.py
```

**Configuration Parameter Management:**
```
MTRR_option.py (learning rate mapping, optimizer, scheduler configuration)
```

**Evaluation and Monitoring:**
```
quality_index.py (PSNR/SSIM/LMSE/NCC) + eval_util.py + early_stop.py
```

### Quick Start
---

**1. Train Model**
```bash
python train.py --epoch 150 --base_lr 1e-4 --scheduler_type plateau
```

**2. Single Dataset Inference**
```bash
python inference.py --ckpt ./model_fit/model_latest.pth --outdir ./infer_outputs
```

**3. Batch Inference**
```bash
python batch_zip_infer.py
```

**4. Performance Analysis**
```bash
python calc_flops_test.py
python calc_throughout.py
```

### Contact
---

If you have any questions or suggestions about this paper, feel free to contact me:

📧 **Email**: luanjingmin@neuq.edu.cn



