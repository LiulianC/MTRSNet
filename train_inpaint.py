                      
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from MTRRNet import MTRREngine
from mask_aware_loss import CoreHaloReconstructionLoss, MaskAwareInpaintLoss
from mtrs_data import CleanMaskBankCorruptionDataset, RandomCleanCorruptionDataset, SupervisedReflectionDataset, mtrs_collate_fn


DEFAULT_TRAIN_MANIFEST = (
    "data/reflection_benchmark/splits/"
    "supervised_clean2pct_scene_v1_evalmask_v1_train.jsonl"
)
DEFAULT_VAL_MANIFEST = (
    "data/reflection_benchmark/splits/"
    "supervised_clean2pct_scene_v1_evalmask_v1_val.jsonl"
)
DEFAULT_CLEAN_TRAIN_MANIFEST = (
    "data/reflection_benchmark/splits/"
    "mtrs_clean_pretrain_clean2pct_v1_train.jsonl"
)
DEFAULT_CLEAN_VAL_MANIFEST = (
    "data/reflection_benchmark/splits/"
    "mtrs_clean_pretrain_clean2pct_v1_val.jsonl"
)
DEFAULT_MASK_BANK_TRAIN_MANIFEST = (
    "data/reflection_benchmark/splits/"
    "supervised_clean2pct_scene_v1_evalmask_v1_256sq_letterbox_v2_train.jsonl"
)
DEFAULT_MASK_BANK_VAL_MANIFEST = (
    "data/reflection_benchmark/splits/"
    "supervised_clean2pct_scene_v1_evalmask_v1_256sq_letterbox_v2_val.jsonl"
)
DEFAULT_RESULTS_ROOT = Path("outputs")
DEFAULT_RUN_ID = "mtrs_improved_evalmask_v1_20260521"


def parse_args():
    parser = argparse.ArgumentParser(
        description="MTRS-improved manifest-backed supervised/corruption training."
    )
    parser.add_argument("--mode", choices=("supervised", "corruption", "clean_mask_corruption"), default="supervised")
    parser.add_argument("--manifest", default=None, help="Backward-compatible alias for --train-manifest.")
    parser.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--clean-train-manifest", default=DEFAULT_CLEAN_TRAIN_MANIFEST)
    parser.add_argument("--clean-val-manifest", default=DEFAULT_CLEAN_VAL_MANIFEST)
    parser.add_argument("--mask-bank-train-manifest", default=DEFAULT_MASK_BANK_TRAIN_MANIFEST)
    parser.add_argument("--mask-bank-val-manifest", default=DEFAULT_MASK_BANK_VAL_MANIFEST)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0, help="0 means no explicit update cap.")
    parser.add_argument(
        "--max-wall-time-sec",
        type=float,
        default=0.0,
        help="Stop training gracefully after this many seconds. 0 disables the wall-time cap.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-mask-ratio", type=float, default=0.02)
    parser.add_argument("--max-mask-ratio", type=float, default=0.20)
    parser.add_argument("--mask-bank-scale-min", type=float, default=0.75)
    parser.add_argument("--mask-bank-scale-max", type=float, default=1.35)
    parser.add_argument("--mask-bank-rotate-degrees", type=float, default=20.0)
    parser.add_argument("--mask-bank-shift-ratio", type=float, default=0.20)
    parser.add_argument("--halo-kernel", type=int, default=15)
    parser.add_argument("--core-loss-weight", type=float, default=1.0)
    parser.add_argument("--halo-loss-weight", type=float, default=0.5)
    parser.add_argument("--identity-loss-weight", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Backward-compatible train limit.")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("val_mask_psnr", "val_core_halo_psnr", "val_halo_psnr", "val_psnr", "val_loss"),
        default="val_mask_psnr",
    )
    parser.add_argument("--val-interval-updates", type=int, default=100)
    parser.add_argument("--sample-interval-updates", type=int, default=100)
    parser.add_argument(
        "--save-interval-updates",
        type=int,
        default=0,
        help="Also overwrite the latest checkpoint every N updates. 0 disables mid-epoch saves.",
    )
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--run-mode", default="scratch_train")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def read_jsonl_count(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_csv_row(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_state_file(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def load_pretrained(model, path, device, optimizer=None):
    if path is None:
        return 0, 0, None
    state = load_state_file(path, device)
    net_state = state.get("netG_T", state.get("state_dict", state))
    if any(str(k).startswith("netG_T.") for k in net_state.keys()):
        net_state = {k.replace("netG_T.", "", 1): v for k, v in net_state.items()}
    missing, unexpected = model.netG_T.load_state_dict(net_state, strict=False)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    start_epoch = int(state.get("epoch", -1)) + 1 if isinstance(state, dict) else 0
    global_step = int(state.get("global_step", 0)) if isinstance(state, dict) else 0
    print(f"loaded checkpoint: {path}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return start_epoch, global_step, state


def build_dataset(args, manifest_path, split, limit):
    if args.mode == "supervised":
        return SupervisedReflectionDataset(manifest_path=manifest_path, split=split, limit=limit)
    if args.mode == "clean_mask_corruption":
        clean_manifest = args.clean_train_manifest if split == "train" else args.clean_val_manifest
        mask_bank_manifest = args.mask_bank_train_manifest if split == "train" else args.mask_bank_val_manifest
        return CleanMaskBankCorruptionDataset(
            clean_manifest_path=clean_manifest,
            mask_bank_manifest_path=mask_bank_manifest,
            split=split,
            limit=limit,
            scale_min=args.mask_bank_scale_min,
            scale_max=args.mask_bank_scale_max,
            rotate_degrees=args.mask_bank_rotate_degrees,
            shift_ratio=args.mask_bank_shift_ratio,
            halo_kernel=args.halo_kernel,
            deterministic=(split != "train"),
        )
    return RandomCleanCorruptionDataset(
        manifest_path=manifest_path,
        split=split,
        min_ratio=args.min_mask_ratio,
        max_ratio=args.max_mask_ratio,
        limit=limit,
    )


def compute_loss(loss_fn, visuals):
    pred_list = visuals["fake_Ts"]
    refl_list = visuals["fake_Rs"]
    target = visuals["T"]
    raw = visuals["Ic"]
    mask = visuals["c_map"]
    weights = (0.5, 0.5, 0.5, 1.0)
    total = None
    final_values = None
    for weight, pred, refl in zip(weights, pred_list, refl_list):
        values = loss_fn(pred, target, raw, mask, refl, visuals["R"])
        loss = values[-1]
        total = loss * weight if total is None else total + loss * weight
        final_values = values
    loss_table, region_loss, detail_loss, ssim_loss, boundary_loss, _ = final_values
    return loss_table, region_loss, detail_loss, ssim_loss, boundary_loss, total


def compute_clean_mask_loss(loss_fn, visuals, batch):
    pred_list = visuals["fake_Ts"]
    target = visuals["T"]
    raw = visuals["Ic"]
    mask = batch.get("M_core", visuals["c_map"]).to(target.device)
    halo = batch.get("M_halo")
    if halo is not None:
        halo = halo.to(target.device)
    weights = (0.5, 0.5, 0.5, 1.0)
    total = None
    final_values = None
    for weight, pred in zip(weights, pred_list):
        values = loss_fn(pred, target, raw, mask, halo_mask=halo)
        loss = values[-1]
        total = loss * weight if total is None else total + loss * weight
        final_values = values
    loss_table, core_loss, halo_loss, identity_loss, halo_l1, _ = final_values
    return loss_table, core_loss, halo_loss, identity_loss, halo_l1, total


def dataset_label(args):
    if args.mode == "clean_mask_corruption":
        return "mtrs_clean_pretrain_clean2pct_v1_plus_evalmask_v1_256sq_letterbox_v2_mask_bank"
    if args.mode == "supervised":
        return "supervised_clean2pct_scene_v1_evalmask_v1"
    return Path(args.train_manifest).stem


def masked_mean(value, mask, eps=1e-8):
    if value.ndim == 4 and mask.ndim == 4 and mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand_as(value)
    denom = mask.sum().clamp_min(eps)
    return (value * mask).sum() / denom


def psnr_from_mse(mse, eps=1e-8):
    mse = float(mse)
    if math.isnan(mse):
        return math.nan
    if mse <= eps:
        return math.inf
    return -10.0 * math.log10(mse)


def region_mse(sq, mask):
    if float(mask.sum().detach().cpu()) <= 0.0:
        return math.nan
    return float(masked_mean(sq, mask).detach().cpu())


def region_mae(abs_err, mask):
    if float(mask.sum().detach().cpu()) <= 0.0:
        return math.nan
    return float(masked_mean(abs_err, mask).detach().cpu())


def collect_batch_metrics(pred, target, mask, fov):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mask = (mask > 0.5).float()
    fov = (fov > 0.5).float()
    if fov.shape[-2:] != pred.shape[-2:]:
        fov = torch.nn.functional.interpolate(fov, size=pred.shape[-2:], mode="nearest")
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = torch.nn.functional.interpolate(mask, size=pred.shape[-2:], mode="nearest")
    full = fov
    mask_region = (mask * fov).clamp(0, 1)
    nonmask = ((1.0 - mask) * fov).clamp(0, 1)
    err = pred - target
    sq = err.pow(2)
    abs_err = err.abs()
    full_mse = region_mse(sq, full)
    mask_mse = region_mse(sq, mask_region)
    nonmask_mse = region_mse(sq, nonmask)
    metrics = {
        "val_loss_proxy": float(abs_err.mean().detach().cpu()),
        "val_psnr": psnr_from_mse(full_mse),
        "val_mask_psnr": psnr_from_mse(mask_mse),
        "val_mask_mae": region_mae(abs_err, mask_region),
        "val_nonmask_psnr": psnr_from_mse(nonmask_mse),
        "val_nonmask_mae": region_mae(abs_err, nonmask),
        "mask_mean": float(mask.mean().detach().cpu()),
    }
    return metrics


def collect_clean_mask_batch_metrics(pred, target, mask, halo, fov):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mask = (mask > 0.5).float()
    halo = (halo > 0.5).float() if halo is not None else torch.zeros_like(mask)
    fov = (fov > 0.5).float()
    for name, region in (("mask", mask), ("halo", halo), ("fov", fov)):
        if region.shape[-2:] != pred.shape[-2:]:
            resized = torch.nn.functional.interpolate(region, size=pred.shape[-2:], mode="nearest")
            if name == "mask":
                mask = resized
            elif name == "halo":
                halo = resized
            else:
                fov = resized
    core = (mask * fov).clamp(0, 1)
    halo_region = (halo * fov * (1.0 - mask)).clamp(0, 1)
    core_halo = ((core + halo_region).clamp(0, 1) * fov).clamp(0, 1)
    nonmask = ((1.0 - core_halo) * fov).clamp(0, 1)
    err = pred - target
    sq = err.pow(2)
    abs_err = err.abs()
    metrics = {
        "val_loss_proxy": float(abs_err.mean().detach().cpu()),
        "val_psnr": psnr_from_mse(region_mse(sq, fov)),
        "val_mask_psnr": psnr_from_mse(region_mse(sq, core)),
        "val_mask_mae": region_mae(abs_err, core),
        "val_halo_psnr": psnr_from_mse(region_mse(sq, halo_region)),
        "val_halo_mae": region_mae(abs_err, halo_region),
        "val_core_halo_psnr": psnr_from_mse(region_mse(sq, core_halo)),
        "val_core_halo_mae": region_mae(abs_err, core_halo),
        "val_nonmask_psnr": psnr_from_mse(region_mse(sq, nonmask)),
        "val_nonmask_mae": region_mae(abs_err, nonmask),
        "mask_mean": float(mask.mean().detach().cpu()),
        "halo_mean": float(halo_region.mean().detach().cpu()),
    }
    return metrics


def average_metric_dicts(rows):
    out = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [
            row[key]
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        if values:
            out[key] = sum(values) / len(values)
    return out


def save_sample_grid(batch, visuals, output_path, max_items=4):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inp = visuals["I"].detach().clamp(0, 1).cpu()
    pred = visuals["fake_Ts"][-1].detach().clamp(0, 1).cpu()
    target = visuals["T"].detach().clamp(0, 1).cpu()
    display_mask = batch.get("M_core") if batch.get("M_halo") is not None else visuals["c_map"]
    mask = display_mask.detach().clamp(0, 1).cpu()
    if mask.shape[1] == 1:
        mask = mask.expand(-1, 3, -1, -1)
    halo = batch.get("M_halo")
    if halo is not None:
        halo = halo.detach().clamp(0, 1).cpu()
        if halo.shape[1] == 1:
            halo = halo.expand(-1, 3, -1, -1)
    error = (pred - target).abs().clamp(0, 1)
    n = min(max_items, inp.shape[0])
    if halo is not None:
        tiles = torch.cat([inp[:n], pred[:n], target[:n], mask[:n], halo[:n], error[:n]], dim=0)
    else:
        tiles = torch.cat([inp[:n], pred[:n], target[:n], mask[:n], error[:n]], dim=0)
    save_image(tiles, output_path, nrow=n, padding=2)


def save_checkpoint(path, model, optimizer, args, epoch, global_step, best_metric, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "netG_T": model.netG_T.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "args": vars(args),
            "metadata": metadata or {},
        },
        path,
    )


def validate(model, loader, loss_fn, device, args, epoch, global_step, samples_dir=None):
    model.eval()
    rows = []
    loss_values = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="val", ncols=120, leave=False)):
            model.set_input(batch)
            model.inference()
            visuals = model.get_current_visuals()
            if args.mode == "clean_mask_corruption":
                loss_values.append(float(compute_clean_mask_loss(loss_fn, visuals, batch)[-1].detach().cpu()))
            else:
                loss_values.append(float(compute_loss(loss_fn, visuals)[-1].detach().cpu()))
            pred = visuals["fake_Ts"][-1]
            target = visuals["T"]
            fov = batch["fov_mask"].to(device)
            if args.mode == "clean_mask_corruption":
                mask = batch.get("M_core", visuals["c_map"]).to(device)
                halo = batch.get("M_halo")
                halo = halo.to(device) if halo is not None else None
                rows.append(collect_clean_mask_batch_metrics(pred, target, mask, halo, fov))
            else:
                mask = visuals["c_map"]
                rows.append(collect_batch_metrics(pred, target, mask, fov))
            if idx == 0 and samples_dir is not None:
                save_sample_grid(
                    batch,
                    visuals,
                    Path(samples_dir) / "val" / f"update_{global_step:06d}_grid.png",
                    args.sample_count,
                )
    metrics = average_metric_dicts(rows)
    if loss_values:
        metrics["val_loss"] = sum(loss_values) / len(loss_values)
    metrics["epoch"] = epoch
    metrics["update"] = global_step
    model.train()
    return metrics


def metric_is_better(metric_name, current, best):
    if current is None or math.isnan(float(current)):
        return False
    if best is None:
        return True
    if metric_name == "val_loss":
        return current < best
    return current > best


def write_protocol(args, run_dir, total_params, trainable_params, train_count, val_count):
    if args.mode == "clean_mask_corruption":
        train_manifest = args.clean_train_manifest
        val_manifest = None if args.no_val else args.clean_val_manifest
        notes = (
            "Clean-mask-corruption mode uses clean2pct MAE-pretrain images as targets and "
            "randomly augmented mask-bank masks as core black-out shapes plus a dilated halo "
            "repair band. It does not use pseudo_clean target_path supervision. "
            "Validation uses the same clean image + "
            "mask-bank corruption protocol and selects checkpoints by clean-corruption metrics."
        )
    else:
        train_manifest = args.train_manifest
        val_manifest = args.val_manifest if not args.no_val else None
        notes = (
            "Supervised mode reads input_path, target_path, mask_path, and normal_mask_path "
            "from supervised_clean2pct_scene_v1_evalmask_v1. Mask semantics follow the "
            "revision protocol: pseudo/evaluation masks, not manual annotations."
        )
    protocol = {
        "run_id": args.run_id,
        "method": "MTRS-improved",
        "mode": args.mode,
        "run_mode": args.run_mode,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "clean_train_manifest": args.clean_train_manifest if args.mode == "clean_mask_corruption" else None,
        "clean_val_manifest": args.clean_val_manifest if args.mode == "clean_mask_corruption" and not args.no_val else None,
        "mask_bank_train_manifest": args.mask_bank_train_manifest if args.mode == "clean_mask_corruption" else None,
        "mask_bank_val_manifest": args.mask_bank_val_manifest if args.mode == "clean_mask_corruption" and not args.no_val else None,
        "train_count": train_count,
        "val_count": val_count,
        "input_size": 256,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "selection_metric": args.selection_metric,
        "pretrained": args.pretrained,
        "resume": args.resume,
        "max_epochs": args.epochs,
        "max_updates": args.max_updates,
        "max_wall_time_sec": args.max_wall_time_sec,
        "mask_bank_augmentation": {
            "scale_min": args.mask_bank_scale_min,
            "scale_max": args.mask_bank_scale_max,
            "rotate_degrees": args.mask_bank_rotate_degrees,
            "shift_ratio": args.mask_bank_shift_ratio,
            "halo_kernel": args.halo_kernel,
            "model_repair_mask": "M_repair=M_core+M_halo",
        } if args.mode == "clean_mask_corruption" else None,
        "loss_weights": {
            "core": args.core_loss_weight,
            "halo": args.halo_loss_weight,
            "identity": args.identity_loss_weight,
        } if args.mode == "clean_mask_corruption" else None,
        "pseudo_target_supervision": False if args.mode == "clean_mask_corruption" else None,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "notes": notes,
    }
    path = Path(run_dir) / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def upsert_registry(args, run_dir, status, total_params, trainable_params, best_checkpoint, latest_checkpoint, start_time, end_time, actual_updates, best_metric):
    registry_path = DEFAULT_RESULTS_ROOT / "experiment_registry.csv"
    if not registry_path.exists():
        return
    with registry_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        return

    found = False
    for row in rows:
        if row.get("run_id") == args.run_id:
            target = row
            found = True
            break
    if not found:
        target = {field: "" for field in fieldnames}
        rows.append(target)

    if args.mode == "clean_mask_corruption":
        protocol_group = "clean_mask_corruption"
        dataset_release = "mtrs_clean_pretrain_clean2pct_v1"
        train_split = args.clean_train_manifest
        val_split = "" if args.no_val else args.clean_val_manifest
        stage = "clean_mask_corruption_train_on_clean2pct_evalmask_bank"
        test_split = ""
        notes = (
            "MTRS-improved clean-mask-corruption training; no pseudo target supervision; "
            f"latest update={actual_updates}."
        )
    else:
        protocol_group = "main_supervised"
        dataset_release = "supervised_clean2pct_scene_v1_evalmask_v1"
        train_split = args.train_manifest
        val_split = args.val_manifest if not args.no_val else ""
        stage = f"{args.mode}_train_on_evalmask_v1"
        test_split = "data/reflection_benchmark/splits/supervised_clean2pct_scene_v1_evalmask_v1_test.jsonl"
        notes = f"MTRS-improved {args.mode} training on evalmask_v1; latest update={actual_updates}."

    target.update(
        {
            "registry_version": "v1",
            "run_id": args.run_id,
            "method": "MTRS-improved",
            "method_family": "ours",
            "role": "main_model",
            "stage": stage,
            "run_mode": args.run_mode,
            "protocol_group": protocol_group,
            "dataset_release": dataset_release,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "input_size": "256",
            "output_size": "256",
            "batch_size_train": str(args.batch_size),
            "batch_size_eval": str(args.eval_batch_size),
            "num_workers": str(args.num_workers),
            "checkpoint_source": args.pretrained or args.resume or "none",
            "pretrained_source": "none" if not args.pretrained else args.pretrained,
            "extra_data": "clean2pct_self_supervised" if args.mode == "clean_mask_corruption" else "no",
            "trainable_params": str(trainable_params),
            "total_params": str(total_params),
            "optimizer": "AdamW",
            "lr": str(args.lr),
            "scheduler": "none",
            "max_epochs": str(args.epochs),
            "max_updates": str(args.max_updates or ""),
            "actual_updates": str(actual_updates),
            "selection_metric": args.selection_metric,
            "best_val_metric": "" if best_metric is None else str(best_metric),
            "best_checkpoint_path": str(best_checkpoint) if best_checkpoint else "",
            "final_checkpoint_path": str(latest_checkpoint),
            "prediction_path": str(Path(run_dir) / "predictions" / "test"),
            "metrics_path": str(Path(run_dir) / "test_metrics_summary.csv"),
            "metadata_path": str(Path(run_dir) / "protocol.json"),
            "log_path": str(Path(run_dir) / "train_log.csv"),
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "repo_path": ".",
            "code_version": "workspace_uncommitted",
            "notes": notes,
        }
    )
    with registry_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_training_efficiency(
    args,
    status,
    total_params,
    trainable_params,
    checkpoint_path,
    start_wall,
    end_wall,
    cumulative_updates,
    segment_updates,
    peak_alloc,
    peak_reserved,
):
    path = DEFAULT_RESULTS_ROOT / "training_efficiency.csv"
    if not path.exists():
        return
    elapsed = max(0.0, end_wall - start_wall)
    avg = elapsed / segment_updates if segment_updates else ""
    row = {
        "log_version": "v1",
        "run_id": args.run_id,
        "method": "MTRS-improved",
        "run_mode": args.run_mode,
        "protocol_group": "clean_mask_corruption" if args.mode == "clean_mask_corruption" else "main_supervised",
        "measured_scope": f"{args.mode}_train_until_current_checkpoint",
        "primary_efficiency_comparable": "no",
        "device": args.device,
        "python_env": os.sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": "12.8_runtime",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "input_size": "256",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "precision": "fp32",
        "amp": "no",
        "gradient_accumulation_steps": "1",
        "train_size": args.train_limit or args.limit or "full",
        "max_epochs": args.epochs,
        "max_updates": args.max_updates or "",
        "actual_epochs": "",
        "actual_updates": segment_updates,
        "warmup_updates_excluded": "0",
        "train_wall_time_sec": elapsed,
        "avg_time_per_update_sec": avg,
        "median_time_per_update_sec": "",
        "p90_time_per_update_sec": "",
        "samples_per_sec": (segment_updates * args.batch_size / elapsed) if elapsed > 0 else "",
        "images_per_update": args.batch_size,
        "peak_memory_allocated_gb": peak_alloc,
        "peak_memory_reserved_gb": peak_reserved,
        "optimizer": "AdamW",
        "lr": args.lr,
        "scheduler": "none",
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "total_params": total_params,
        "model_size_mb": Path(checkpoint_path).stat().st_size / 1024**2 if Path(checkpoint_path).exists() else "",
        "checkpoint_source": args.pretrained or args.resume or "none",
        "checkpoint_path": str(checkpoint_path),
        "log_path": str(DEFAULT_RESULTS_ROOT / args.run_id / "train_log.csv"),
        "status": status,
        "notes": (
            "Formal run log; training cost is adaptation/scratch cost disclosure, not inference efficiency. "
            f"Cumulative optimizer updates after this segment: {cumulative_updates}."
        ),
    }
    with path.open("r", encoding="utf-8", newline="") as f:
        fieldnames = next(csv.reader(f))
    write_csv_row(path, fieldnames, row)


def main():
    args = parse_args()
    if args.manifest:
        args.train_manifest = args.manifest
    if args.smoke:
        args.train_limit = args.train_limit or args.limit or max(args.batch_size, 2)
        args.val_limit = min(args.val_limit, max(args.eval_batch_size, 2))
        args.max_updates = args.max_updates or 1
        args.val_interval_updates = 1
        args.sample_interval_updates = 1

    results_root = Path(args.results_root)
    run_dir = results_root / args.run_id
    checkpoint_dir = Path(args.save_dir) if args.save_dir else run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_limit = args.train_limit if args.train_limit is not None else args.limit
    val_workers = args.num_workers if args.val_workers is None else args.val_workers
    train_dataset = build_dataset(args, args.train_manifest, "train", train_limit)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=mtrs_collate_fn,
    )
    val_loader = None
    if not args.no_val:
        val_dataset = build_dataset(args, args.val_manifest, "val", args.val_limit)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=mtrs_collate_fn,
        )
    else:
        val_dataset = None

    device = torch.device(args.device)
    model = MTRREngine(opts=None, device=device)
    if args.mode == "clean_mask_corruption":
        loss_fn = CoreHaloReconstructionLoss(
            core_weight=args.core_loss_weight,
            halo_weight=args.halo_loss_weight,
            identity_weight=args.identity_loss_weight,
            halo_kernel=args.halo_kernel,
        ).to(device)
    else:
        loss_fn = MaskAwareInpaintLoss().to(device)
    optimizer = torch.optim.AdamW(model.netG_T.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    resume_state = None
    if args.resume:
        start_epoch, global_step, resume_state = load_pretrained(model, args.resume, device, optimizer=optimizer)
    elif args.pretrained:
        load_pretrained(model, args.pretrained, device, optimizer=None)
    initial_global_step = global_step

    total_params, trainable_params = count_params(model.netG_T)
    train_count = len(train_dataset)
    val_count = len(val_dataset) if val_dataset is not None else 0
    protocol_path = write_protocol(args, run_dir, total_params, trainable_params, train_count, val_count)
    train_dataset_label = dataset_label(args)

    train_fields = [
        "run_id",
        "method",
        "dataset",
        "stage",
        "epoch",
        "update",
        "lr",
        "train_loss",
        "loss_region",
        "loss_detail",
        "loss_ssim",
        "loss_boundary",
        "time_sec",
        "gpu_mem_mb",
    ]
    val_fields = [
        "run_id",
        "method",
        "dataset",
        "epoch",
        "update",
        "val_loss",
        "val_psnr",
        "val_mask_psnr",
        "val_mask_mae",
        "val_halo_psnr",
        "val_halo_mae",
        "val_core_halo_psnr",
        "val_core_halo_mae",
        "val_nonmask_psnr",
        "val_nonmask_mae",
        "mask_mean",
        "halo_mean",
        "selection_metric",
        "is_best",
        "checkpoint_path",
    ]

    best_metric = resume_state.get("best_metric") if isinstance(resume_state, dict) else None
    if best_metric is not None and not math.isfinite(float(best_metric)):
        best_metric = None
    best_checkpoint = checkpoint_dir / "mtrs_improved_inpaint_best.pth" if best_metric is not None else None
    latest_checkpoint = checkpoint_dir / "mtrs_improved_inpaint_latest.pth"
    start_time = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    start_wall = time.perf_counter()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.train()
    stop_training = False
    end_epoch = start_epoch + args.epochs
    last_epoch = start_epoch
    for epoch in range(start_epoch, end_epoch):
        last_epoch = epoch
        pbar = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}", ncols=120)
        for batch in pbar:
            step_start = time.perf_counter()
            model.set_input(batch)
            model.inference()
            visuals = model.get_current_visuals()
            if args.mode == "clean_mask_corruption":
                loss_table, region_loss, detail_loss, ssim_loss, boundary_loss, loss = compute_clean_mask_loss(
                    loss_fn, visuals, batch
                )
            else:
                loss_table, region_loss, detail_loss, ssim_loss, boundary_loss, loss = compute_loss(loss_fn, visuals)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.netG_T.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            gpu_mem_mb = ""
            if torch.cuda.is_available() and device.type == "cuda":
                gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            elapsed = time.perf_counter() - step_start
            write_csv_row(
                run_dir / "train_log.csv",
                train_fields,
                {
                    "run_id": args.run_id,
                    "method": "MTRS-improved",
                    "dataset": train_dataset_label,
                    "stage": args.mode,
                    "epoch": epoch,
                    "update": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": float(loss.detach().cpu()),
                    "loss_region": float(region_loss.detach().cpu()),
                    "loss_detail": float(detail_loss.detach().cpu()),
                    "loss_ssim": float(ssim_loss.detach().cpu()),
                    "loss_boundary": float(boundary_loss.detach().cpu()),
                    "time_sec": elapsed,
                    "gpu_mem_mb": gpu_mem_mb,
                },
            )

            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach().cpu()):.5f}",
                    "region": f"{float(region_loss.detach().cpu()):.5f}",
                    "mask": f"{float(visuals['c_map'].mean().detach().cpu()):.4f}",
                }
            )

            if args.sample_interval_updates > 0 and global_step % args.sample_interval_updates == 0:
                save_sample_grid(
                    batch,
                    visuals,
                    samples_dir / "train" / f"update_{global_step:06d}_grid.png",
                    args.sample_count,
                )

            should_validate = (
                val_loader is not None
                and args.val_interval_updates > 0
                and global_step % args.val_interval_updates == 0
            )
            if should_validate:
                metrics = validate(model, val_loader, loss_fn, device, args, epoch, global_step, samples_dir)
                current = metrics.get(args.selection_metric)
                is_best = metric_is_better(args.selection_metric, current, best_metric)
                if is_best:
                    best_metric = current
                    best_checkpoint = checkpoint_dir / "mtrs_improved_inpaint_best.pth"
                    save_checkpoint(
                        best_checkpoint,
                        model,
                        optimizer,
                        args,
                        epoch,
                        global_step,
                        best_metric,
                        {"validation": metrics, "protocol": str(protocol_path)},
                    )
                write_csv_row(
                    run_dir / "val_metrics.csv",
                    val_fields,
                    {
                        "run_id": args.run_id,
                        "method": "MTRS-improved",
                        "dataset": train_dataset_label,
                        "epoch": epoch,
                        "update": global_step,
                        "val_loss": metrics.get("val_loss", ""),
                        "val_psnr": metrics.get("val_psnr", ""),
                        "val_mask_psnr": metrics.get("val_mask_psnr", ""),
                        "val_mask_mae": metrics.get("val_mask_mae", ""),
                        "val_halo_psnr": metrics.get("val_halo_psnr", ""),
                        "val_halo_mae": metrics.get("val_halo_mae", ""),
                        "val_core_halo_psnr": metrics.get("val_core_halo_psnr", ""),
                        "val_core_halo_mae": metrics.get("val_core_halo_mae", ""),
                        "val_nonmask_psnr": metrics.get("val_nonmask_psnr", ""),
                        "val_nonmask_mae": metrics.get("val_nonmask_mae", ""),
                        "mask_mean": metrics.get("mask_mean", ""),
                        "halo_mean": metrics.get("halo_mean", ""),
                        "selection_metric": args.selection_metric,
                        "is_best": "true" if is_best else "false",
                        "checkpoint_path": str(best_checkpoint) if is_best else "",
                    },
                )

            if args.save_interval_updates > 0 and global_step % args.save_interval_updates == 0:
                save_checkpoint(
                    latest_checkpoint,
                    model,
                    optimizer,
                    args,
                    epoch,
                    global_step,
                    best_metric,
                    {"protocol": str(protocol_path)},
                )

            if args.max_updates and global_step >= args.max_updates:
                stop_training = True
                break
            if args.max_wall_time_sec > 0 and (time.perf_counter() - start_wall) >= args.max_wall_time_sec:
                print(
                    f"Reached max wall time {args.max_wall_time_sec:.1f}s at update {global_step}; "
                    "saving checkpoint and stopping gracefully.",
                    flush=True,
                )
                stop_training = True
                break
        save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            args,
            epoch,
            global_step,
            best_metric,
            {"protocol": str(protocol_path)},
        )
        if stop_training:
            break

    if val_loader is not None and global_step > 0 and (
        args.val_interval_updates <= 0 or global_step % args.val_interval_updates != 0
    ):
        metrics = validate(model, val_loader, loss_fn, device, args, last_epoch, global_step, samples_dir)
        current = metrics.get(args.selection_metric)
        is_best = metric_is_better(args.selection_metric, current, best_metric)
        if is_best:
            best_metric = current
            best_checkpoint = checkpoint_dir / "mtrs_improved_inpaint_best.pth"
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                args,
                last_epoch,
                global_step,
                best_metric,
                {"validation": metrics, "protocol": str(protocol_path)},
            )
        write_csv_row(
            run_dir / "val_metrics.csv",
            val_fields,
            {
                "run_id": args.run_id,
                "method": "MTRS-improved",
                "dataset": train_dataset_label,
                "epoch": last_epoch,
                "update": global_step,
                "val_loss": metrics.get("val_loss", ""),
                "val_psnr": metrics.get("val_psnr", ""),
                "val_mask_psnr": metrics.get("val_mask_psnr", ""),
                "val_mask_mae": metrics.get("val_mask_mae", ""),
                "val_halo_psnr": metrics.get("val_halo_psnr", ""),
                "val_halo_mae": metrics.get("val_halo_mae", ""),
                "val_core_halo_psnr": metrics.get("val_core_halo_psnr", ""),
                "val_core_halo_mae": metrics.get("val_core_halo_mae", ""),
                "val_nonmask_psnr": metrics.get("val_nonmask_psnr", ""),
                "val_nonmask_mae": metrics.get("val_nonmask_mae", ""),
                "mask_mean": metrics.get("mask_mean", ""),
                "halo_mean": metrics.get("halo_mean", ""),
                "selection_metric": args.selection_metric,
                "is_best": "true" if is_best else "false",
                "checkpoint_path": str(best_checkpoint) if is_best else "",
            },
        )

    save_checkpoint(
        latest_checkpoint,
        model,
        optimizer,
        args,
        last_epoch,
        global_step,
        best_metric,
        {"protocol": str(protocol_path)},
    )
    end_wall = time.perf_counter()
    end_time = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() and device.type == "cuda" else ""
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() and device.type == "cuda" else ""
    status = "completed_smoke" if args.smoke else "completed_training_segment"
    upsert_registry(
        args,
        run_dir,
        status,
        total_params,
        trainable_params,
        best_checkpoint,
        latest_checkpoint,
        start_time,
        end_time,
        global_step,
        best_metric,
    )
    append_training_efficiency(
        args,
        status,
        total_params,
        trainable_params,
        latest_checkpoint,
        start_wall,
        end_wall,
        global_step,
        global_step - initial_global_step,
        peak_alloc,
        peak_reserved,
    )
    print(f"run_id: {args.run_id}")
    print(f"updates: {global_step}")
    print(f"latest checkpoint: {latest_checkpoint}")
    print(f"best checkpoint: {best_checkpoint}")
    print(f"best {args.selection_metric}: {best_metric}")
    print(f"protocol: {protocol_path}")


if __name__ == "__main__":
    main()
