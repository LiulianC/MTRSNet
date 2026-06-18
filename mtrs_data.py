import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DATA_ROOT = Path("data/reflection_benchmark")
RELEASE_ROOT = DATA_ROOT / "releases"
MAIN_SUPERVISED_RELEASE = "supervised_clean2pct_scene_v1_evalmask_v1"
FALLBACK_SUPERVISED_RELEASE = "supervised_clean2pct_scene_v1"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mtrs_collate_fn(batch):
                                                                                    
    if not batch:
        return {}
    manifest_rows = [item.get("row", {}) for item in batch]
    collatable = []
    for item in batch:
        filtered = {key: value for key, value in item.items() if key != "row"}
        collatable.append(filtered)
    out = default_collate(collatable)
    out["row"] = manifest_rows
    return out


def load_rgb_tensor(path, size=256):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return TF.to_tensor(img)


def load_mask_tensor(path, size=256):
    img = Image.open(path).convert("L")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    return (TF.to_tensor(img) > 0.5).float()


def estimate_fov_mask(image, threshold=0.02):
                                 
    return (image.mean(dim=0, keepdim=True) > threshold).float()


def _draw_rect(mask, y0, x0, h, w):
    _, H, W = mask.shape
    y1 = min(H, max(0, y0 + h))
    x1 = min(W, max(0, x0 + w))
    y0 = max(0, min(H, y0))
    x0 = max(0, min(W, x0))
    if y1 > y0 and x1 > x0:
        mask[:, y0:y1, x0:x1] = 1.0


def random_fov_corruption_mask(fov_mask, min_ratio=0.02, max_ratio=0.20, max_rects=12, rng=None):
                       
    rng = rng or random
    device = fov_mask.device
    fov = fov_mask > 0.5
    _, H, W = fov.shape
    fov_count = int(fov.sum().item())
    if fov_count <= 0:
        return torch.zeros_like(fov_mask)

    target = int(fov_count * rng.uniform(min_ratio, max_ratio))
    target = max(1, target)
    mask = torch.zeros_like(fov_mask)
    ys, xs = torch.where(fov[0])
    if ys.numel() == 0:
        return mask

    attempts = 0
    while int((mask * fov_mask).sum().item()) < target and attempts < max_rects * 20:
        attempts += 1
        idx = rng.randrange(ys.numel())
        cy = int(ys[idx].item())
        cx = int(xs[idx].item())
        area = rng.randint(max(16, target // (max_rects * 2)), max(32, target // 2))
        aspect = math.exp(rng.uniform(math.log(0.35), math.log(2.8)))
        h = max(4, int(round(math.sqrt(area / aspect))))
        w = max(4, int(round(math.sqrt(area * aspect))))
        y0 = cy - h // 2
        x0 = cx - w // 2
        _draw_rect(mask, y0, x0, h, w)
        mask = mask * fov_mask

    if int(mask.sum().item()) == 0:
        idx = rng.randrange(ys.numel())
        _draw_rect(mask, int(ys[idx].item()) - 8, int(xs[idx].item()) - 8, 16, 16)
        mask = mask * fov_mask
    return mask.to(device=device)


def crop_or_pad_mask(mask, size, rng=None):
    rng = rng or random
    _, h, w = mask.shape
    if h > size:
        top = rng.randint(0, h - size)
        mask = mask[:, top : top + size, :]
    elif h < size:
        pad_top = rng.randint(0, size - h)
        pad_bottom = size - h - pad_top
        mask = F.pad(mask, (0, 0, pad_top, pad_bottom))

    _, h, w = mask.shape
    if w > size:
        left = rng.randint(0, w - size)
        mask = mask[:, :, left : left + size]
    elif w < size:
        pad_left = rng.randint(0, size - w)
        pad_right = size - w - pad_left
        mask = F.pad(mask, (pad_left, pad_right, 0, 0))
    return mask


def shift_mask(mask, max_shift_ratio=0.20, rng=None):
    rng = rng or random
    _, h, w = mask.shape
    max_dy = int(round(h * max_shift_ratio))
    max_dx = int(round(w * max_shift_ratio))
    dy = rng.randint(-max_dy, max_dy) if max_dy > 0 else 0
    dx = rng.randint(-max_dx, max_dx) if max_dx > 0 else 0
    out = torch.zeros_like(mask)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = mask[:, src_y0:src_y1, src_x0:src_x1]
    return out


def augment_mask_bank_mask(
    mask,
    size=256,
    rng=None,
    scale_min=0.75,
    scale_max=1.35,
    rotate_degrees=20.0,
    shift_ratio=0.20,
):
    rng = rng or random
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] != 1:
        mask = mask.amax(dim=0, keepdim=True)

    if rng.random() < 0.5:
        mask = torch.flip(mask, dims=[2])
    if rng.random() < 0.2:
        mask = torch.flip(mask, dims=[1])

    scale = rng.uniform(scale_min, scale_max)
    scaled_size = max(4, int(round(size * scale)))
    mask = F.interpolate(
        mask.unsqueeze(0),
        size=(scaled_size, scaled_size),
        mode="nearest",
    ).squeeze(0)
    mask = crop_or_pad_mask(mask, size, rng)

    if rotate_degrees > 0:
        angle = rng.uniform(-rotate_degrees, rotate_degrees)
        mask = TF.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0.0)

    if shift_ratio > 0:
        mask = shift_mask(mask, shift_ratio, rng)

    return (mask > 0.5).float()


def make_halo_mask(mask, kernel_size=15):
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    pad = kernel_size // 2
    core = mask.clamp(0.0, 1.0)
    dilated = F.max_pool2d(core, kernel_size=kernel_size, stride=1, padding=pad)
    halo = (dilated - core).clamp(0.0, 1.0)
    return halo.squeeze(0)


class CleanPretrainDataset(Dataset):
    def __init__(self, manifest_path=None, split="train", size=256, limit=None):
        if manifest_path is None:
            manifest_path = RELEASE_ROOT / "mtrs_clean_pretrain_clean2pct_v1" / f"{split}.jsonl"
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        if limit is not None:
            self.rows = self.rows[: int(limit)]
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_rgb_tensor(row["input_path"], self.size)
        fov = estimate_fov_mask(image)
        return {
            "input": image,
            "target_t": image.clone(),
            "target_r": torch.zeros_like(image),
            "M_core": torch.zeros(1, image.shape[-2], image.shape[-1]),
            "fov_mask": fov,
            "fn": row.get("sample_id", Path(row["input_path"]).stem),
            "row": row,
        }


class RandomCleanCorruptionDataset(Dataset):
    def __init__(
        self,
        manifest_path=None,
        split="train",
        size=256,
        min_ratio=0.02,
        max_ratio=0.20,
        limit=None,
    ):
        if manifest_path is None:
            manifest_path = RELEASE_ROOT / MAIN_SUPERVISED_RELEASE / f"{split}.jsonl"
            if not Path(manifest_path).exists():
                manifest_path = RELEASE_ROOT / FALLBACK_SUPERVISED_RELEASE / f"{split}.jsonl"
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        if limit is not None:
            self.rows = self.rows[: int(limit)]
        self.size = size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clean_path = row.get("target_path") or row.get("input_path")
        clean = load_rgb_tensor(clean_path, self.size)
        fov = self._load_fov(row, clean)
        mask = random_fov_corruption_mask(fov, self.min_ratio, self.max_ratio)
        corrupted = clean * (1.0 - mask)
        return {
            "input": corrupted,
            "target_t": clean,
            "target_r": clean - corrupted,
            "M_core": mask,
            "fov_mask": fov,
            "fn": row.get("sample_id", Path(clean_path).stem),
            "row": row,
        }

    def _load_fov(self, row, clean):
        normal_path = row.get("normal_mask_path")
        if normal_path:
            path = Path(normal_path)
            if path.exists():
                return load_mask_tensor(path, self.size)
        return estimate_fov_mask(clean)


class CleanMaskBankCorruptionDataset(Dataset):
    def __init__(
        self,
        clean_manifest_path=None,
        mask_bank_manifest_path=None,
        split="train",
        size=256,
        limit=None,
        mask_limit=None,
        scale_min=0.75,
        scale_max=1.35,
        rotate_degrees=20.0,
        shift_ratio=0.20,
        halo_kernel=15,
        deterministic=False,
        seed=20260525,
    ):
        if clean_manifest_path is None:
            clean_manifest_path = RELEASE_ROOT / "mtrs_clean_pretrain_clean2pct_v1" / f"{split}.jsonl"
        if mask_bank_manifest_path is None:
            mask_bank_manifest_path = RELEASE_ROOT / MAIN_SUPERVISED_RELEASE / f"{split}.jsonl"

        self.clean_manifest_path = Path(clean_manifest_path)
        self.mask_bank_manifest_path = Path(mask_bank_manifest_path)
        self.rows = read_jsonl(self.clean_manifest_path)
        self.mask_rows = [row for row in read_jsonl(self.mask_bank_manifest_path) if row.get("mask_path")]
        if limit is not None:
            self.rows = self.rows[: int(limit)]
        if mask_limit is not None:
            self.mask_rows = self.mask_rows[: int(mask_limit)]
        if not self.mask_rows:
            raise ValueError(f"No mask_path entries found in mask bank: {self.mask_bank_manifest_path}")
        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rotate_degrees = rotate_degrees
        self.shift_ratio = shift_ratio
        self.halo_kernel = halo_kernel
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rng = random.Random(self.seed + int(index)) if self.deterministic else random
        clean_row = self.rows[index]
        clean_path = clean_row.get("input_path") or clean_row.get("target_path")
        clean = load_rgb_tensor(clean_path, self.size)
        fov = self._load_fov(clean_row, clean)
        mask_row = self._select_mask_row(index, rng)
        mask = load_mask_tensor(mask_row["mask_path"], self.size)
        mask = augment_mask_bank_mask(
            mask,
            size=self.size,
            rng=rng,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            rotate_degrees=self.rotate_degrees,
            shift_ratio=self.shift_ratio,
        )
        mask = (mask * fov).clamp(0.0, 1.0)
        if int(mask.sum().item()) == 0:
            mask = random_fov_corruption_mask(fov, min_ratio=0.02, max_ratio=0.12, rng=rng)
        halo = (make_halo_mask(mask, self.halo_kernel) * fov * (1.0 - mask)).clamp(0.0, 1.0)
        repair_mask = (mask + halo).clamp(0.0, 1.0)
        corrupted = clean * (1.0 - repair_mask)
        row = {
            "clean_row": clean_row,
            "mask_row": {
                "sample_id": mask_row.get("sample_id"),
                "dataset": mask_row.get("dataset"),
                "mask_path": mask_row.get("mask_path"),
                "mask_source": mask_row.get("mask_source"),
            },
            "protocol_note": (
                "clean image masked with augmented mask-bank shape plus halo repair band; "
                "loss is computed on core and halo only; no pseudo target supervision"
            ),
        }
        return {
            "input": corrupted,
            "target_t": clean,
            "target_r": clean - corrupted,
            "M_core": mask,
            "M_halo": halo,
            "M_repair": repair_mask,
            "fov_mask": fov,
            "fn": clean_row.get("sample_id", Path(clean_path).stem),
            "row": row,
        }

    def _select_mask_row(self, index, rng):
        if self.deterministic:
            return self.mask_rows[index % len(self.mask_rows)]
        return self.mask_rows[rng.randrange(len(self.mask_rows))]

    def _load_fov(self, row, clean):
        normal_path = row.get("normal_mask_path")
        if normal_path:
            path = Path(normal_path)
            if path.exists():
                return load_mask_tensor(path, self.size)
        return estimate_fov_mask(clean)


class SupervisedReflectionDataset(Dataset):
    def __init__(self, manifest_path=None, split="test", size=256, limit=None):
        if manifest_path is None:
            manifest_path = RELEASE_ROOT / MAIN_SUPERVISED_RELEASE / f"{split}.jsonl"
            if not Path(manifest_path).exists():
                manifest_path = RELEASE_ROOT / FALLBACK_SUPERVISED_RELEASE / f"{split}.jsonl"
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        if limit is not None:
            self.rows = self.rows[: int(limit)]
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        inp = load_rgb_tensor(row["input_path"], self.size)
        target = load_rgb_tensor(row["target_path"], self.size)
        mask = self._load_mask(row, inp, target)
        fov = self._load_fov(row, target)
        return {
            "input": inp,
            "target_t": target,
            "target_r": (inp - target) * mask,
            "M_core": mask,
            "fov_mask": fov,
            "fn": row.get("sample_id", Path(row["input_path"]).stem),
            "row": row,
        }

    def _load_mask(self, row, inp, target):
        mask_path = row.get("mask_path")
        if mask_path:
            path = Path(mask_path)
            if path.exists():
                return load_mask_tensor(path, self.size)
        residual = (inp - target).abs().amax(dim=0, keepdim=True)
        return (residual > 0.05).float() * estimate_fov_mask(inp)

    def _load_fov(self, row, target):
        normal_path = row.get("normal_mask_path")
        if normal_path:
            path = Path(normal_path)
            if path.exists():
                return load_mask_tensor(path, self.size)
        return estimate_fov_mask(target)


def masked_region_mean(value, mask, eps=1e-6):
    if value.ndim == 4 and mask.ndim == 4 and mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(eps)
