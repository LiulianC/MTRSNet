# MTRSNet

Minimal public release for MTRSNet training and inference.

This repository contains the model code, manifest-backed dataset loaders, training and inference entrypoints, and the dataset-construction helper `delight.py`. The shared inference checkpoint is released separately on Google Drive. The fully constructed benchmark is intentionally not uploaded because it is too large for Google Drive, so readers must rebuild the training data locally from the source datasets below.

## Demo

![Demo](assert/demo.gif)

## Architecture

![Architecture](assert/arch.png)

## Released Checkpoint

- MAE clean pretrain stage3 checkpoint: [mtrs_improved_pretrain_latest.pth](https://drive.google.com/file/d/1uVJ_gNi5j0GbBm3pdPEJybNzX_6bXFqf/view?usp=sharing)

Use this checkpoint directly for inference, or pass it to `train_inpaint.py` as `--pretrained` for clean-mask-corruption fine-tuning.

## External Datasets

The constructed benchmark is derived from the datasets below. The release only shares the code and the manifest recipes; the large constructed dataset itself must be rebuilt locally.

| Dataset | Link | Notes |
| --- | --- | --- |
| HyperKavsir | [HyperKavsir](https://github.com/researchmm/STTN) | Endoscopic image dataset used as a source for clean / pseudo-clean construction. |
| MouseData | [MouseData](https://drive.google.com/file/d/11ZyY300UrTOw7slB9LtF69BE3M64b1k7/view?usp=sharing) | Mouse tissue dataset used as a source for paired or pseudo-clean construction. |
| VASST-desmoke | [VASST-desmoke](https://ieee-dataport.org/documents/vivo-laparoscopic-image-desmoking-dataset) | Paired laparoscopic de-smoking dataset. The full release has more than 3000 smoky / smoke-free pairs. |
| CholecT50 | [Dataset page](http://camma.u-strasbg.fr/datasets) and [GitHub](https://github.com/CAMMA-public/cholect50) | Endoscopic surgical video dataset used as a source for synthetic highlight construction. |

## Repository Files

- `MTRRNet.py`: model definition and engine.
- `MTRR_token_modules.py`, `MTRR_RD_modules.py`, `vmamba.py`: network building blocks.
- `mtrs_data.py`: JSONL manifest datasets and mask-bank corruption utilities.
- `mask_aware_loss.py`: supervised and mask-aware reconstruction losses.
- `pretrain_mae.py`: clean-image MAE-style pretraining entrypoint.
- `train_inpaint.py`: supervised, clean-corruption, and clean-mask-corruption training entrypoint.
- `export_predictions.py`: checkpoint prediction export entrypoint.
- `delight.py`: pseudo reflection-mask generator used when building datasets from raw source images.

## Environment

Recommended runtime:

- Python 3.10 or 3.11
- CUDA-enabled PyTorch that matches your local CUDA toolkit / `nvcc`
- `mamba-ssm` and `causal-conv1d` built for the same CUDA stack
- Linux with an NVIDIA GPU

Install PyTorch and the CUDA stack first, then install the Python packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```

If `mamba-ssm` or `causal-conv1d` do not have a matching wheel for your CUDA build, follow the official Mamba repository installation notes and install them against the existing CUDA-enabled PyTorch build. The Mamba repo documents the source-install path with `--no-build-isolation`; that is the route to use when a direct wheel install is not enough.

`delight.py` needs OpenCV. `opencv-python-headless` is included in the requirements file so the helper can run on headless machines as well.

Quick environment sanity check:

```bash
python -V
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
nvcc --version
```

## Dataset Construction

The code reads JSONL manifests under `data/reflection_benchmark/`.

The required fields depend on the protocol:

- `input_path`: input image path.
- `target_path`: clean target image path when paired supervision exists.
- `mask_path`: reflection / highlight mask path when available.
- `normal_mask_path`: optional valid-field-of-view mask.
- `sample_id`: stable sample identifier.

The supported protocols are:

- Clean MAE pretraining: JSONL rows with `input_path` only.
- Supervised reflection training: paired rows with `input_path` and `target_path`, plus `mask_path` and `normal_mask_path` when available.
- Clean-mask-corruption training: one clean manifest for source images and one mask-bank manifest for corruption shapes.
- Random clean corruption training: clean images only.

If a source dataset does not already provide a reflection mask, run `delight.py` first to generate a pseudo mask:

```bash
python delight.py /path/to/image_or_dir [divisor=10] [erode_px=3]
```

This writes two files next to each source image:

- `*_dd_grey.png`: double-divided grayscale map.
- `*_ref_mask.png`: pseudo reflection mask.

Use the generated mask path in your manifest when you need a `mask_path`.

The example manifests under `data/reflection_benchmark/splits/` show the released protocol layout and field names. They are meant to be rebuilt locally rather than uploaded as a giant archive.

## Inference

Inference uses the supervised-style manifest loader. The manifest should provide `input_path` and `target_path`, and may also include `mask_path` and `normal_mask_path` if you want to reuse the released benchmark layout.

Inputs:

- `--manifest`: paired or pseudo-paired JSONL manifest
- `--checkpoint`: trained `netG_T` checkpoint, including the released pretrain stage3 weight or a fine-tuned weight
- `--split`: logical split name used for output naming

Outputs:

- PNG predictions in `--output-dir`, or in `outputs/<run-id>/predictions/<split>/` when `--output-dir` is omitted
- `<split>_prediction_export_metadata.json` next to the prediction folder

Example:

```bash
python export_predictions.py \
  --manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_test.jsonl \
  --checkpoint /path/to/mtrs_improved_pretrain_latest.pth \
  --output-dir outputs/predictions/test
```

## Training

Training requires the reader to reconstruct the dataset first using the manifest format described above.

### Clean MAE Pretraining

Inputs:

- `--manifest`: clean-image JSONL rows with `input_path`

Outputs:

- `--save-dir/mtrs_improved_pretrain_latest.pth`
- `--output-dir/pretrain_log.csv`
- optional mid-run overwrite checkpoints if `--save-interval-updates` is set

Example:

```bash
python pretrain_mae.py \
  --manifest data/reflection_benchmark/splits/mtrs_clean_pretrain_clean2pct_v1_train.jsonl \
  --save-dir checkpoints/pretrain_stage3 \
  --output-dir outputs/pretrain_stage3 \
  --epochs 30 \
  --batch-size 48
```

### Formal Training

`train_inpaint.py` accepts three modes:

- `supervised`: `--train-manifest` and `--val-manifest` must point to paired manifests with `input_path` and `target_path`. `mask_path` and `normal_mask_path` are used when present.
- `clean_mask_corruption`: `--clean-train-manifest` and `--clean-val-manifest` provide clean source images, while `--mask-bank-train-manifest` and `--mask-bank-val-manifest` provide the corruption mask bank.
- `corruption`: a clean or paired manifest is used to synthesize random FOV corruption masks.

Inputs:

- `--results-root` or `--save-dir`
- one or more manifests depending on `--mode`
- `--pretrained` when starting clean-mask-corruption fine-tuning from the released stage3 checkpoint

Outputs:

- `results-root/run-id/checkpoints/mtrs_improved_inpaint_latest.pth`
- `results-root/run-id/checkpoints/mtrs_improved_inpaint_best.pth` when validation selects a best checkpoint
- `results-root/run-id/train_log.csv`
- `results-root/run-id/val_metrics.csv`
- `results-root/run-id/protocol.json`
- `results-root/run-id/samples/train/*.png` and `results-root/run-id/samples/val/*.png`

Example clean-mask-corruption run:

```bash
python train_inpaint.py \
  --mode clean_mask_corruption \
  --clean-train-manifest data/reflection_benchmark/splits/mtrs_clean_pretrain_clean2pct_v1_train.jsonl \
  --clean-val-manifest data/reflection_benchmark/splits/mtrs_clean_pretrain_clean2pct_v1_val.jsonl \
  --mask-bank-train-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_256sq_letterbox_v2_train.jsonl \
  --mask-bank-val-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_256sq_letterbox_v2_val.jsonl \
  --pretrained /path/to/mtrs_improved_pretrain_latest.pth \
  --results-root outputs \
  --epochs 30 \
  --batch-size 1
```

Example supervised run:

```bash
python train_inpaint.py \
  --mode supervised \
  --train-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_train.jsonl \
  --val-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_val.jsonl \
  --results-root outputs
```

## Notes

- Do not commit checkpoints, datasets, generated samples, or training outputs to Git.
- The shared Google Drive checkpoint is intended for inference and fine-tuning; the large constructed dataset is not uploaded.
- If you add a new source dataset, keep the manifest field names consistent with `mtrs_data.py`.
