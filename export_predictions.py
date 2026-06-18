                      
import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from MTRRNet import MTRREngine
from mtrs_data import SupervisedReflectionDataset, mtrs_collate_fn


DEFAULT_TEST_MANIFEST = (
    "data/reflection_benchmark/splits/"
    "supervised_clean2pct_scene_v1_evalmask_v1_test.jsonl"
)
DEFAULT_RUN_ID = "mtrs_improved_evalmask_v1_20260521"
DEFAULT_RESULTS_ROOT = Path("outputs")


def parse_args():
    parser = argparse.ArgumentParser(description="Export MTRS-improved predictions for the unified evaluator.")
    parser.add_argument("--manifest", default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    net_state = state.get("netG_T", state.get("state_dict", state))
    if any(str(k).startswith("netG_T.") for k in net_state.keys()):
        net_state = {k.replace("netG_T.", "", 1): v for k, v in net_state.items()}
    missing, unexpected = model.netG_T.load_state_dict(net_state, strict=False)
    print(f"loaded checkpoint: {path}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def as_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [str(value)]


def save_tensor_png(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = TF.to_pil_image(tensor.detach().cpu().clamp(0, 1))
    image.save(path)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_ROOT / args.run_id / "predictions" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SupervisedReflectionDataset(manifest_path=args.manifest, split=args.split, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=mtrs_collate_fn,
    )

    device = torch.device(args.device)
    model = MTRREngine(opts=None, device=device).eval()
    load_checkpoint(model, args.checkpoint, device)

    written = 0
    skipped = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(loader, desc="export", ncols=120):
            model.set_input(batch)
            model.inference()
            pred = model.get_current_visuals()["fake_Ts"][-1].clamp(0, 1)
            names = as_list(batch["fn"])
            for idx, sample_id in enumerate(names):
                output_path = out_dir / f"{sample_id}.png"
                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                save_tensor_png(pred[idx], output_path)
                written += 1

    elapsed = time.perf_counter() - start
    metadata = {
        "run_id": args.run_id,
        "method": "MTRS-improved",
        "manifest": args.manifest,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "output_dir": str(out_dir),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "written": written,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "images_per_sec": written / elapsed if elapsed > 0 else None,
        "python": os.sys.executable,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == "cuda" else str(device),
        "prediction_size": "256x256",
    }
    metadata_path = out_dir.parent / f"{args.split}_prediction_export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"predictions: {out_dir}")
    print(f"written: {written}, skipped: {skipped}, elapsed_sec: {elapsed:.3f}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
