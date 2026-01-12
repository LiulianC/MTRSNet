import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset.new_dataset1 import *
from torch.utils.data import ConcatDataset
import os
from PIL import Image

def adaptive_gamma_match_tensor(src_tensor, ref_tensor):
    """
    自适应 Gamma 校正 (Tensor 版本)
    - 依据亮度（近似）自动调整 gamma，使暗图提亮，亮图适度压暗
    - 不改变色相与饱和度（在 RGB 空间上施加幂次）
    输入: src_tensor (C,H,W) [0,1], ref_tensor (C,H,W) [0,1]
    返回: 校正后的 Tensor (C,H,W) [0,1]
    """
    if src_tensor.dim() == 2:
        src_tensor = src_tensor.unsqueeze(0)
    if ref_tensor.dim() == 2:
        ref_tensor = ref_tensor.unsqueeze(0)

    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=src_tensor.device, dtype=src_tensor.dtype)
    src_lum = torch.einsum('chw,c->hw', src_tensor.float(), weights)
    ref_lum = torch.einsum('chw,c->hw', ref_tensor.float(), weights)

    eps = 1e-6
    src_med = torch.median(src_lum.flatten()).clamp_min(eps)
    ref_med = torch.median(ref_lum.flatten()).clamp_min(eps)

    src_m = src_med.clamp_min(eps)
    ref_m = ref_med.clamp_min(eps)
    gamma = torch.log(ref_m) / torch.log(src_m)
    gamma = torch.clamp(gamma, 0.5, 1.5)

    corrected = torch.pow(src_tensor.float().clamp(0, 1), gamma)
    return corrected.clamp(0, 1)

def hist_match_channel_tensor(src_channel, ref_channel, bins=256):
    """
    单通道直方图匹配 (Tensor 版本) - 简化实现
    输入: 单通道 Tensor (H,W) [0,1]
    返回: 匹配后的单通道 Tensor (H,W) [0,1]
    """
    src_channel = torch.clamp(src_channel, 0, 1)
    ref_channel = torch.clamp(ref_channel, 0, 1)
    
    src_uint8 = (src_channel * 255.0).clamp(0, 255).round().to(torch.uint8)
    ref_uint8 = (ref_channel * 255.0).clamp(0, 255).round().to(torch.uint8)
    
    device = src_channel.device
    
    src_hist = torch.histc(src_uint8.float(), bins=256, min=0, max=255)
    ref_hist = torch.histc(ref_uint8.float(), bins=256, min=0, max=255)
    
    eps = 1e-8
    src_hist += eps
    ref_hist += eps
    
    src_cdf = src_hist.cumsum(0) / src_hist.sum()
    ref_cdf = ref_hist.cumsum(0) / ref_hist.sum()
    
    lut = torch.zeros(256, device=device)
    for i in range(256):
        j = torch.searchsorted(ref_cdf, src_cdf[i], right=True)
        j = torch.clamp(j, 0, 255)
        lut[i] = j
    
    matched_uint8 = lut[src_uint8.long()].clamp(0, 255)
    
    return matched_uint8.float() / 255.0


def _srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(x):
    a = 0.055
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1/2.4)) - a)

def _rgb_to_xyz(rgb):
    R, G, B = rgb[0], rgb[1], rgb[2]
    X = 0.4124564*R + 0.3575761*G + 0.1804375*B
    Y = 0.2126729*R + 0.7151522*G + 0.0721750*B
    Z = 0.0193339*R + 0.1191920*G + 0.9503041*B
    return torch.stack([X, Y, Z], dim=0)

def _xyz_to_rgb(xyz):
    X, Y, Z = xyz[0], xyz[1], xyz[2]
    R =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
    G = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    B =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
    return torch.stack([R, G, B], dim=0)

def _xyz_to_lab(xyz):
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[0] / Xn
    y = xyz[1] / Yn
    z = xyz[2] / Zn
    eps = 216/24389  
    kappa = 24389/27 
    def f(t):
        return torch.where(t > eps, t.pow(1/3), (kappa * t + 16.)/116.)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116.*fy - 16.
    a = 500.*(fx - fy)
    b = 200.*(fy - fz)
    return torch.stack([L, a, b], dim=0)

def _lab_to_xyz(lab):
    L, a, b = lab[0], lab[1], lab[2]
    fy = (L + 16.) / 116.
    fx = fy + a / 500.
    fz = fy - b / 200.
    eps = 216/24389
    kappa = 24389/27
    def finv(t):
        return torch.where(t ** 3 > eps, t**3, (116.*t - 16.)/kappa)
    xr, yr, zr = finv(fx), finv(fy), finv(fz)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X, Y, Z = xr * Xn, yr * Yn, zr * Zn
    return torch.stack([X, Y, Z], dim=0)

def _srgb_to_lab(img):
    lin = _srgb_to_linear(img.clamp(0,1))
    xyz = _rgb_to_xyz(lin)
    lab = _xyz_to_lab(xyz)
    return lab

def _lab_to_srgb(lab):
    xyz = _lab_to_xyz(lab)
    rgb_lin = _xyz_to_rgb(xyz)
    rgb = _linear_to_srgb(rgb_lin)
    return rgb.clamp(0,1)

def _hist_match_lab_channel(src_ch, ref_ch, low=-128.0, high=127.0):
    src_u8 = ((src_ch.clamp(low, high) - low) * (255.0 / (high - low))).round().clamp(0,255).to(torch.uint8)
    ref_u8 = ((ref_ch.clamp(low, high) - low) * (255.0 / (high - low))).round().clamp(0,255).to(torch.uint8)

    src_hist = torch.histc(src_u8.float(), bins=256, min=0, max=255) + 1e-8
    ref_hist = torch.histc(ref_u8.float(), bins=256, min=0, max=255) + 1e-8
    src_cdf = src_hist.cumsum(0) / src_hist.sum()
    ref_cdf = ref_hist.cumsum(0) / ref_hist.sum()

    lut = torch.zeros(256, device=src_ch.device)
    for i in range(256):
        j = torch.searchsorted(ref_cdf, src_cdf[i], right=True)
        j = torch.clamp(j, 0, 255)
        lut[i] = j

    matched_u8 = lut[src_u8.long()].clamp(0,255)
    matched = matched_u8.float() * (high - low) / 255.0 + low
    return matched

def hist_match_rgb_tensor(src_tensor, ref_tensor, bins=256):
    """
    Lab 空间的颜色匹配（更自然）：
    - L 通道做自适应 Gamma（提亮/压暗）以靠近参考亮度
    - a/b 通道做直方图匹配（使色彩分布接近参考）
    输入: src_tensor/ref_tensor: (3,H,W) [0,1]
    返回: 匹配后的 Tensor (3,H,W) [0,1]
    """
    src = src_tensor.clamp(0,1)
    ref = ref_tensor.clamp(0,1)

    src_lab = _srgb_to_lab(src)
    ref_lab = _srgb_to_lab(ref)

    L_src = (src_lab[0] / 100.0).clamp(0,1).unsqueeze(0)
    L_ref = (ref_lab[0] / 100.0).clamp(0,1).unsqueeze(0)
    L_corr = adaptive_gamma_match_tensor(L_src, L_ref).squeeze(0).clamp(0,1) * 100.0

    a_corr = _hist_match_lab_channel(src_lab[1], ref_lab[1])
    b_corr = _hist_match_lab_channel(src_lab[2], ref_lab[2])



    out_lab = torch.stack([L_corr, a_corr, b_corr], dim=0)
    out_rgb = _lab_to_srgb(out_lab).clamp(0,1)

    return out_rgb

def hist_match_batch_tensor(src_batch, ref_batch, bins=256):
    """
    批量 RGB 直方图匹配 (Tensor 版本)
    输入: src_batch (N,C,H,W) [0,1], ref_batch (N,C,H,W) [0,1]
    返回: 匹配后的 Tensor (N,C,H,W) [0,1]
    """
    matched_batch = []
    for i in range(src_batch.size(0)):
        matched = hist_match_rgb_tensor(
            src_batch[i], 
            ref_batch[i], 
            bins=bins
        )
        matched_batch.append(matched)
    return torch.stack(matched_batch)





import torch
import numpy as np
from PIL import Image, ImageOps
import cv2  

def histogram_equalization_lab(img_tensor, 
                              enhance_l=True, 
                              enhance_a=False, 
                              enhance_b=False,
                              clip_limit_l=0.5,
                              clip_limit_a=0.4,
                              clip_limit_b=0.4,
                              tile_size=(8,8)):
    """
    在Lab颜色空间中进行温和的自适应直方图均衡化
    输入: 
        img_tensor: torch.Tensor (C, H, W) 值范围 [-1, 1]
        enhance_l: 是否增强亮度通道(L)
        enhance_a: 是否增强颜色通道(a)
        enhance_b: 是否增强颜色通道(b)
        clip_limit_l: L通道的对比度限制阈值(推荐1.0-3.0)
        clip_limit_a: a通道的对比度限制阈值(推荐1.0-2.0)
        clip_limit_b: b通道的对比度限制阈值(推荐1.0-2.0)
        tile_size: 局部处理块大小
    输出: torch.Tensor (C, H, W) 值范围 [-1, 1]
    """
    img_np = img_tensor.mul(0.5).add(0.5).clamp(0, 1).mul(255).byte()
    img_np = img_np.permute(1, 2, 0).cpu().numpy()  
    
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    if enhance_l:
        clahe_l = cv2.createCLAHE(clipLimit=clip_limit_l, tileGridSize=tile_size)
        l_channel = clahe_l.apply(l_channel)
    
    if enhance_a:
        clahe_a = cv2.createCLAHE(clipLimit=clip_limit_a, tileGridSize=tile_size)
        a_channel = clahe_a.apply(a_channel)
    
    if enhance_b:
        clahe_b = cv2.createCLAHE(clipLimit=clip_limit_b, tileGridSize=tile_size)
        b_channel = clahe_b.apply(b_channel)
    
    img_lab_eq = cv2.merge([l_channel, a_channel, b_channel])
    
    img_rgb_eq = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB)
    
    img_eq_tensor = torch.from_numpy(img_rgb_eq.astype(np.float32) / 255.0)
    img_eq_tensor = img_eq_tensor.permute(2, 0, 1)  
    img_eq_tensor = img_eq_tensor.mul(2).sub(1)  
    
    return img_eq_tensor









if __name__ == "__main__":
    save_dir = "/home/hostname/hostname-MTRRNetv2/hist_match_results"
    os.makedirs(save_dir, exist_ok=True)

    test_data_dir1 = '/home/hostname/hostname-MTRRVideo/data/tissue_real'
    test_data1 = DSRTestDataset(
        datadir=test_data_dir1, 
        fns='/home/hostname/hostname-MTRRVideo/data/tissue_real_index/eval1.txt', 
        enable_transforms=False, 
        if_align=True, 
        real=True, 
        HW=[256,256], 
        size=200, 
        SamplerSize=False, 
        color_match=True
    )    
    train_loader = torch.utils.data.DataLoader(
        test_data1, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0, 
        drop_last=False, 
        pin_memory=True
    )

    def save_image(tensor, path):
        """
        将张量保存为图像文件
        输入: 
            tensor: (C,H,W) 图像张量 [0,1]
            path: 保存路径
        """
        tensor = tensor.mul(255).byte().cpu()
        if tensor.dim() == 3 and tensor.size(0) == 3:
            img = Image.fromarray(tensor.permute(1, 2, 0).numpy(), 'RGB')
        elif tensor.dim() == 3 and tensor.size(0) == 1:
            img = Image.fromarray(tensor.squeeze(0).numpy(), 'L')
        else:
            img = Image.fromarray(tensor.numpy())
        
        img.save(path)

    src_tensor = test_data1[0]['input']  
    ref_tensor = test_data1[0]['target_t']  
    
    print(f"Source min: {src_tensor.min().item()}, max: {src_tensor.max().item()}")
    print(f"Reference min: {ref_tensor.min().item()}, max: {ref_tensor.max().item()}")
    
    matched_tensor = hist_match_rgb_tensor(src_tensor, ref_tensor)
    
    print(f"Matched min: {matched_tensor.min().item()}, max: {matched_tensor.max().item()}")

    save_image(src_tensor, os.path.join(save_dir, "source.png"))
    save_image(ref_tensor, os.path.join(save_dir, "reference.png"))
    save_image(matched_tensor, os.path.join(save_dir, "matched.png"))

    for batch_idx, batch in enumerate(train_loader):
        src_batch = batch['input']  
        ref_batch = batch['target_t']  
        
        print(f"Batch {batch_idx} Source min: {src_batch.min().item()}, max: {src_batch.max().item()}")
        print(f"Batch {batch_idx} Reference min: {ref_batch.min().item()}, max: {ref_batch.max().item()}")
        
        matched_batch = hist_match_batch_tensor(src_batch, ref_batch)
        print(f"Batch {batch_idx} Matched min: {matched_batch.min().item()}, max: {matched_batch.max().item()}")
        print(f"Batch {batch_idx} Matched Shape:", matched_batch.shape)
        
        batch_dir = os.path.join(save_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        
        for i in range(src_batch.size(0)):
            save_image(src_batch[i], os.path.join(batch_dir, f"source_{i}.png"))
            save_image(ref_batch[i], os.path.join(batch_dir, f"reference_{i}.png"))
            save_image(matched_batch[i], os.path.join(batch_dir, f"matched_{i}.png"))
        
        break

    print(f"所有图像已保存到: {save_dir}")