# MTRSNet

Minimal training and inference release for MTRSNet.

This repository contains the model definition, manifest-backed datasets, MAE-style clean pretraining, supervised/corruption fine-tuning, and prediction export code. Checkpoints and datasets are intentionally not tracked in Git.

## Files

- `MTRRNet.py`: MTRSNet model and training/inference engine.
- `MTRR_token_modules.py`, `MTRR_RD_modules.py`, `vmamba.py`: network building blocks.
- `mtrs_data.py`: JSONL manifest datasets and mask-bank corruption utilities.
- `mask_aware_loss.py`: supervised and mask-aware reconstruction losses.
- `pretrain_mae.py`: clean-image MAE-style pretraining entrypoint.
- `train_inpaint.py`: supervised, clean-corruption, and clean-mask-corruption training entrypoint.
- `export_predictions.py`: checkpoint prediction export entrypoint.

## Environment

Install PyTorch and CUDA dependencies for your machine first, then install the Python packages listed in `requirements.txt`.

The model uses Mamba/VMamba-style blocks. A CUDA environment with compatible `mamba-ssm`, `causal-conv1d`, and Triton builds is recommended.

## Dataset Layout

The code reads JSONL manifests. By default it looks under:

```text
data/reflection_benchmark/
```

You can either place/symlink the benchmark there or pass explicit manifest paths with command-line flags.

Expected row fields vary by protocol, but the common fields are:

- `input_path`: input or clean image path.
- `target_path` or `target_t_path`: clean transmission target path.
- `target_r_path`: reflection/residual target path when available.
- `M_core_path`, `mask_path`, or equivalent mask fields when available.
- `sample_id`: optional stable sample identifier.

## Clean MAE Pretraining

```bash
python pretrain_mae.py \
  --manifest data/reflection_benchmark/splits/mtrs_clean_pretrain_clean2pct_v1_train.jsonl \
  --save-dir checkpoints/pretrain_stage3 \
  --output-dir outputs/pretrain_stage3 \
  --epochs 30 \
  --batch-size 48
```

The release checkpoint used for downstream fine-tuning is not stored in Git. Keep it in external storage and pass its path with `--pretrained`.

## Fine-Tuning

Clean-mask-corruption training with a clean pretrain checkpoint:

```bash
python train_inpaint.py \
  --mode clean_mask_corruption \
  --clean-train-manifest data/reflection_benchmark/splits/mtrs_clean_pretrain_clean2pct_v1_train.jsonl \
  --clean-val-manifest data/reflection_benchmark/splits/mtrs_clean_pretrain_clean2pct_v1_val.jsonl \
  --mask-bank-train-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_256sq_letterbox_v2_train.jsonl \
  --mask-bank-val-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_256sq_letterbox_v2_val.jsonl \
  --pretrained /path/to/mtrs_improved_pretrain_latest.pth \
  --run-id mtrs_clean_mask_corruption \
  --results-root outputs \
  --epochs 30 \
  --batch-size 1
```

Supervised training:

```bash
python train_inpaint.py \
  --mode supervised \
  --train-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_train.jsonl \
  --val-manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_val.jsonl \
  --results-root outputs
```

## Export Predictions

```bash
python export_predictions.py \
  --manifest data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_test.jsonl \
  --checkpoint /path/to/mtrs_improved_inpaint_best.pth \
  --output-dir outputs/predictions/test
```

## Checkpoints And Data

Do not commit checkpoints, datasets, generated samples, or training outputs. Use external storage such as Google Drive for large artifacts.
