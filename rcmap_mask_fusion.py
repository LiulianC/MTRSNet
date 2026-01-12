import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os

def apply_rcmap_mask_and_skip(test_imgs, test_fake_Ts, test_rcmaps, threshold=10):
    device = test_imgs.device
    test_fake_T3 = test_fake_Ts.to(device)
    test_rcmaps = test_rcmaps.to(device)
    
    mask = (test_rcmaps > threshold/255).float()  
    
    mask_rgb = mask.repeat(1, 3, 1, 1)  
    
    AdditionSkip = test_fake_T3 * mask_rgb + test_imgs * (1 - mask_rgb)


    return AdditionSkip, mask

if __name__ == "__main__":
    batch_size = 4
    H, W = 256, 256
    
    test_imgs = torch.randn(batch_size, 3, H, W)  
    test_fake_Ts = [
        torch.randn(batch_size, 3, H, W) for _ in range(4)  
    ]
    test_rcmaps = torch.randn(batch_size, 1, H, W) * 50  
    
    AdditionSkip = apply_rcmap_mask_and_skip(test_imgs, test_fake_Ts, test_rcmaps, threshold=30)
    
    print(f"输入图像形状: {test_imgs.shape}")
    print(f"输出图像形状: {AdditionSkip.shape}")
    print(f"掩码应用完成!")