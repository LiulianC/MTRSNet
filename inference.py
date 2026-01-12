import os
import argparse
import datetime
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from torch.utils.data import ConcatDataset
from dataset.new_dataset1 import HyperKDataset_Test,DSRTestDataset
from MTRRNet import MTRREngine
from MTRR_option import build_train_opts
from torchvision.utils import make_grid
from util.color_enhance import histogram_equalization_lab,hist_match_batch_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for MTRRNet on HyperKDataset_Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path (.pth). If not set, use opts.model_path")
    parser.add_argument("--outdir", type=str, default="./infer_outputs", help="output directory for predictions")
    parser.add_argument("--save-reflection", default=False, help="also save predicted reflection layer")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _as_list(x) -> List[str]:
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return [str(x)]


def main():
    args = parse_args()

    opts = build_train_opts([])

    device = torch.device(args.device)


    tissue_dir = '/home/hostname/hostname-MTRRVideo/data/tissue_real'
    tissue_data = DSRTestDataset(datadir=tissue_dir,fns='/home/hostname/hostname-MTRRVideo/data/tissue_real_index/train1.txt',size=800, enable_transforms=True, unaligned_transforms=False, if_align=True,real=True, HW=[256,256], SamplerSize=True, color_match=False)
    test_data_dir1 = '/home/hostname/hostname-MTRRVideo/data/tissue_real'
    test_data1 = DSRTestDataset(datadir=test_data_dir1, fns='/home/hostname/hostname-MTRRVideo/data/tissue_real_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=200, SamplerSize=False, color_match=False)
    dataset = ConcatDataset([tissue_data,test_data1])
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)



    model = MTRREngine(opts, device)
    model.eval()

    ckpt_path = args.ckpt or getattr(opts, "model_path", None)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[i] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    if isinstance(state, dict) and "netG_T" in state:
        net_state = {k.replace("netG_T.", ""): v for k, v in state["netG_T"].items()}
        model.netG_T.load_state_dict(net_state, strict=True)
        print("[i] Loaded netG_T from training state")
    else:
        model.netG_T.load_state_dict(state, strict=True)
        print("[i] Loaded netG_T from raw state_dict")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.outdir, f"output_infer_{timestamp}")
    ensure_dir(out_dir)


    saved = 0
    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), ncols=120, desc="Inferring")
        for num, batch in enumerate(pbar):
            inputs = batch["input"].to(device, non_blocking=True)
            zeros = torch.zeros_like(inputs)
            feed = {"input": inputs, "target_t": zeros, "target_r": zeros}

            model.set_input(feed)
            model.inference()

            visuals = model.get_current_visuals()

            fake_T = visuals["fake_Ts"][-1]   

            
            fake_T_eq = hist_match_batch_tensor(fake_T,inputs)

            t_path = os.path.join(out_dir, f"{num:04d}-grid_fakeT.png")
            grid_fakeT = make_grid(fake_T_eq, nrow=4, padding=0, normalize=True)
            save_image(grid_fakeT, t_path)

            if args.save_reflection:
                r_path = os.path.join(out_dir, f"{num:04d}-grid_fakeR.png")
                fake_R = visuals["fake_Rs"][-1]   
                grid_fakeR = make_grid(fake_R, nrow=4, padding=0, normalize=True)
                save_image(grid_fakeR, r_path)

            i_path = os.path.join(out_dir, f"{num:04d}-grid_input.png")
            grid_input = make_grid(inputs, nrow=4, padding=0, normalize=True)
            save_image(grid_input, i_path)

            pbar.set_postfix({"saved": saved})
        pbar.close()

    print(f"[✓] Done. Saved {saved} images to: {out_dir}")



if __name__ == "__main__":
    main()
