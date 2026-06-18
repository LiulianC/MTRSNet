import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from MTRRNet import MTRRNet
from mtrs_data import CleanPretrainDataset


def infer_square_grid(num_tokens):
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer square token grid from {num_tokens} tokens")
    return side, side


def conv_grid_size(height, width, patch_size, stride, padding):
    h = (height + 2 * padding - patch_size) // stride + 1
    w = (width + 2 * padding - patch_size) // stride + 1
    if h <= 0 or w <= 0:
        raise ValueError(
            f"Invalid patch geometry for input {height}x{width}: "
            f"patch_size={patch_size}, stride={stride}, padding={padding}"
        )
    return h, w


def build_random_token_mask(batch_size, num_tokens, min_ratio, max_ratio, device):
    if min_ratio < 0 or max_ratio > 1 or min_ratio > max_ratio:
        raise ValueError(f"Invalid mask ratio range: [{min_ratio}, {max_ratio}]")

    ratios = torch.empty(batch_size, device=device).uniform_(min_ratio, max_ratio)
    num_masks = torch.round(ratios * num_tokens).long().clamp(min=1, max=num_tokens)
    noise = torch.rand(batch_size, num_tokens, device=device)
    ids = torch.argsort(noise, dim=1)
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    for b in range(batch_size):
        mask[b, ids[b, : int(num_masks[b].item())]] = True
    return mask, ratios, num_masks


def image_to_overlap_patches(image, patch_size=6, stride=4, padding=1):
    patches = F.unfold(image, kernel_size=patch_size, stride=stride, padding=padding)
    return patches.transpose(1, 2).contiguous()


def masked_patch_pixel_mse(pred_patches, target_image, token_mask, patch_size=6, stride=4, padding=1):
    target_patches = image_to_overlap_patches(
        target_image,
        patch_size=patch_size,
        stride=stride,
        padding=padding,
    )
    if pred_patches.shape != target_patches.shape:
        raise ValueError(
            f"Prediction patches {tuple(pred_patches.shape)} do not match "
            f"target patches {tuple(target_patches.shape)}"
        )
    token_mask = token_mask.to(device=pred_patches.device, dtype=torch.bool)
    masked_pred = pred_patches[token_mask]
    masked_target = target_patches[token_mask]
    if masked_pred.numel() == 0:
        raise ValueError("No masked patches available for MAE loss")
    return (masked_pred - masked_target).pow(2).mean()


class MultiScaleMAEDecoder(nn.Module):
                                                                               

    def __init__(
        self,
        input_dims=(96, 96, 192, 384),
        decoder_dim=96,
        depth=8,
        num_heads=4,
        mlp_ratio=4.0,
        patch_size=6,
        out_chans=3,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.proj = nn.ModuleList([nn.Linear(dim, decoder_dim) for dim in input_dims])
        self.level_scale = nn.Parameter(torch.ones(len(input_dims)))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_dim,
                    nhead=num_heads,
                    dim_feedforward=int(decoder_dim * mlp_ratio),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, out_chans * patch_size * patch_size)

    @staticmethod
    def _resize_tokens(tokens, target_grid):
        b, n, c = tokens.shape
        h, w = infer_square_grid(n)
        th, tw = target_grid
        if (h, w) == (th, tw):
            return tokens
        feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        feat = F.interpolate(feat, size=(th, tw), mode="bilinear", align_corners=False)
        return feat.flatten(2).transpose(1, 2).contiguous()

    def forward(self, tokens_list, token_grid):
        if len(tokens_list) != len(self.proj):
            raise ValueError(f"Expected {len(self.proj)} token levels, got {len(tokens_list)}")

        fused = None
        for idx, (tokens, proj) in enumerate(zip(tokens_list, self.proj)):
            x = proj(tokens)
            x = self._resize_tokens(x, token_grid)
            x = x * self.level_scale[idx]
            fused = x if fused is None else fused + x
        x = fused / math.sqrt(len(tokens_list))

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


class MTRSMAEPretrainer(nn.Module):
    def __init__(self, decoder_dim=96, decoder_depth=8, decoder_heads=4, decoder_mlp_ratio=4.0, patch_size=6):
        super().__init__()
        self.netG_T = MTRRNet()
        self.mae_decoder = MultiScaleMAEDecoder(
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            patch_size=patch_size,
        )

    def forward(self, clean, token_mask):
        zero_mask = torch.zeros(
            clean.shape[0],
            1,
            clean.shape[-2],
            clean.shape[-1],
            device=clean.device,
            dtype=clean.dtype,
        )
        repair_input = torch.cat([clean, zero_mask], dim=1)
        tokens_list, token_mask, token_grid = self.netG_T.token_encoder(
            repair_input,
            token_mask=token_mask,
            mask_ratio=0.0,
            return_token_mask=True,
        )
        pred_patches = self.mae_decoder(tokens_list, token_grid)
        return {
            "pred_patches": pred_patches,
            "token_mask": token_mask,
            "token_grid": token_grid,
        }


def write_csv_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Token-level MAE pretraining for MTRS-improved.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0, help="0 means no explicit update cap.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-mask-ratio", type=float, default=0.15)
    parser.add_argument("--max-mask-ratio", type=float, default=0.25)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--patch-padding", type=int, default=1)
    parser.add_argument("--decoder-dim", type=int, default=96)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="checkpoints/MTRS-improved")
    parser.add_argument("--output-dir", default="outputs/pretrain_mae")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--save-interval-updates",
        type=int,
        default=0,
        help="Also overwrite the latest checkpoint every N updates. 0 disables mid-epoch saves.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, args, epoch, global_step):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "netG_T": model.netG_T.state_dict(),
        "mae_decoder": model.mae_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "pretrain_config": {
            "type": "clean_image_token_mae",
            "min_mask_ratio": args.min_mask_ratio,
            "max_mask_ratio": args.max_mask_ratio,
            "patch_size": args.patch_size,
            "patch_stride": args.patch_stride,
            "patch_padding": args.patch_padding,
            "decoder_dim": args.decoder_dim,
            "decoder_depth": args.decoder_depth,
            "decoder_heads": args.decoder_heads,
            "decoder_mlp_ratio": args.decoder_mlp_ratio,
            "prediction_activation": "none",
            "target_normalization": "none",
            "loss": "raw_pixel_mse_on_masked_overlap_patches",
        },
    }
    torch.save(ckpt, path)


def load_resume(path, model, optimizer, device):
    state = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.netG_T.load_state_dict(state["netG_T"], strict=False)
    decoder_missing, decoder_unexpected = model.mae_decoder.load_state_dict(
        state["mae_decoder"], strict=False
    )
    if "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"resumed: {path}")
    print(f"netG_T missing={len(missing)} unexpected={len(unexpected)}")
    print(f"mae_decoder missing={len(decoder_missing)} unexpected={len(decoder_unexpected)}")
    return int(state.get("epoch", -1)) + 1, int(state.get("global_step", 0))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    limit = args.limit
    if args.smoke and limit is None:
        limit = max(args.batch_size, 2)
    if args.smoke and args.max_updates == 0:
        args.max_updates = 1

    dataset = CleanPretrainDataset(manifest_path=args.manifest, split="train", limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = MTRSMAEPretrainer(
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        patch_size=args.patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    start_epoch = 0
    if args.resume:
        start_epoch, global_step = load_resume(args.resume, model, optimizer, device)

    stop_training = False
    log_path = Path(args.output_dir) / "pretrain_log.csv"
    latest_path = Path(args.save_dir) / "mtrs_improved_pretrain_latest.pth"

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"pretrain epoch {epoch}", ncols=120)
        for batch in pbar:
            clean = batch["input"].to(device, non_blocking=True)
            ht, wt = conv_grid_size(
                clean.shape[-2],
                clean.shape[-1],
                args.patch_size,
                args.patch_stride,
                args.patch_padding,
            )
            token_mask, mask_ratios, num_masks = build_random_token_mask(
                clean.shape[0],
                ht * wt,
                args.min_mask_ratio,
                args.max_mask_ratio,
                clean.device,
            )

            out = model(clean, token_mask)
            loss = masked_patch_pixel_mse(
                out["pred_patches"],
                clean,
                out["token_mask"],
                patch_size=args.patch_size,
                stride=args.patch_stride,
                padding=args.patch_padding,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            mask_mean = float(mask_ratios.mean().detach().cpu().item())
            mask_count_mean = float(num_masks.float().mean().detach().cpu().item())
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "mask_ratio": f"{mask_mean:.4f}",
                    "mask_count": f"{mask_count_mean:.1f}",
                }
            )
            write_csv_row(
                log_path,
                {
                    "epoch": epoch,
                    "update": global_step,
                    "loss": f"{loss.item():.10f}",
                    "mask_ratio_mean": f"{mask_mean:.8f}",
                    "mask_count_mean": f"{mask_count_mean:.4f}",
                    "token_grid_h": ht,
                    "token_grid_w": wt,
                    "batch_size": clean.shape[0],
                    "lr": args.lr,
                },
            )

            if args.max_updates and global_step >= args.max_updates:
                stop_training = True
                break

            if args.save_interval_updates > 0 and global_step % args.save_interval_updates == 0:
                save_checkpoint(latest_path, model, optimizer, args, epoch, global_step)

        save_checkpoint(latest_path, model, optimizer, args, epoch, global_step)
        if stop_training:
            break

    print(f"saved: {latest_path}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
