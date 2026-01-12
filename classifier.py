
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import os
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import argparse
from torch.utils.data import ConcatDataset




class PretrainedConvNext(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext, self).__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.head = nn.Linear(768, 6) 
    def forward(self, x):
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        
        out = self.model(cls_input)
        out = self.head(out)
        return out
    



class PretrainedConvNext_e2e(nn.Module):
    def __init__(self, model_name='convnext_base', pretrained=True):
        super(PretrainedConvNext_e2e, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(768, 6)

    def forward(self, x):
        with torch.no_grad():
            cls_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        out = self.model(cls_input)
        out = self.head(out)
        alpha, beta = out[..., :3].unsqueeze(-1).unsqueeze(-1), out[..., 3:].unsqueeze(-1).unsqueeze(-1)
        out = alpha * x + beta
        return out



def ssim_loss(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    B = pred_np.shape[0]
    total_ssim = 0.0
    for i in range(B):
        p = np.transpose(pred_np[i], (1, 2, 0))  
        t = np.transpose(target_np[i], (1, 2, 0))
        total_ssim += ssim(p, t, channel_axis=2, data_range=1.0, win_size=11)
    return 1 - (total_ssim / B)


def visualize(input, pred, target, epoch, save_dir='./cls/vis'):
    os.makedirs(save_dir, exist_ok=True)
    input = torch.clamp(input, 0, 1)
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    for i in range((pred.size(0))):  
        grid = torch.cat([input[i], pred[i], target[i]], dim=-1)  
        save_image(grid, os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.png'))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts = parser.parse_args()
opts.batch_size = 32
opts.sampler_size1 = 0
opts.sampler_size2 = 0
opts.sampler_size3 = 800
opts.test_size = [200, 0, 0]
opts.epochs = 10
opts.model_path='./cls/cls_models/latest.pth'  
opts.model_path=None  
current_lr = 1e-5
opts.num_workers = 0
opts.shuffle = True

if __name__ == "__main__":
    from dataset.new_dataset1 import *  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedConvNext_e2e('convnext_small_in22k').to(device)
    if opts.model_path is not None:
        print(f"Loading model from {opts.model_path}")
        model.load_state_dict(torch.load(opts.model_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    mse_fn = nn.MSELoss()

    fit_datadir = '/home/hostname/hostname-RDNet1/dataset/laparoscope_gen'
    fit_data = DSRTestDataset(datadir=fit_datadir, fns='/home/hostname/hostname-RDNet1/dataset/laparoscope_gen_index/train1.txt',size=opts.sampler_size1, enable_transforms=False,if_align=True,real=False, HW=[256,256])

    tissue_gen = '/home/hostname/hostname-MTRRVideo/data/tissue_gen'
    tissue_gen_data = DSRTestDataset(datadir=tissue_gen, fns='/home/hostname/hostname-MTRRVideo/data/tissue_gen_index/train1.txt',size=opts.sampler_size2, enable_transforms=False,if_align=True,real=False, HW=[256,256])

    tissue_dir = '/home/hostname/hostname-MTRRVideo/data/tissue_real'
    tissue_data = DSRTestDataset(datadir=tissue_dir,fns='/home/hostname/hostname-MTRRVideo/data/tissue_real_index/train1.txt',size=opts.sampler_size3, enable_transforms=True,if_align=True,real=False, HW=[256,256])

    VOCroot = "/home/hostname/hostname-RDNet1/dataset/VOC2012"
    VOCjson_file = "/home/hostname/hostname-RDNet1/dataset/VOC2012/VOC_results_list.json"
    VOCdataset = VOCJsonDataset(VOCroot, VOCjson_file, size=400, enable_transforms=False, HW=[256, 256])

    HyperKroot = "/home/hostname/hostname-MTRRNetv2/data/EndoData"
    HyperKJson = "/home/hostname/hostname-MTRRNetv2/data/EndoData/test.json"
    HyperK_data = HyperKDataset(root=HyperKroot, json_path=HyperKJson, start=343, end=369, size=12800, enable_transforms=True, unaligned_transforms=False, if_align=True, HW=[256,256], flag=None)

    train_data = ConcatDataset([fit_data, tissue_gen_data, tissue_data, VOCdataset, HyperK_data])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size, shuffle=opts.shuffle, num_workers = opts.num_workers, drop_last=False, pin_memory=True)



    test_data_dir1 = '/home/hostname/hostname-MTRRVideo/data/tissue_real'
    test_data1 = DSRTestDataset(datadir=test_data_dir1, fns='/home/hostname/hostname-MTRRVideo/data/tissue_real_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[0])

    test_data_dir2 = '/home/hostname/hostname-MTRRVideo/data/hyperK_000'
    test_data2 = TestDataset(datadir=test_data_dir2, fns='/home/hostname/hostname-MTRRVideo/data/hyperK_000_list.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[1])

    test_data_dir3 = '/home/hostname/hostname-RDNet1/dataset/laparoscope_gen'
    test_data3 = DSRTestDataset(datadir=test_data_dir3, fns='/home/hostname/hostname-RDNet1/dataset/laparoscope_gen_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[2])

    VOCroot1 = "/home/hostname/hostname-RDNet1/dataset/VOC2012"
    VOCjson_file1 = "/home/hostname/hostname-RDNet1/dataset/VOC2012/VOC_results_list.json"
    VOCdataset1 = VOCJsonDataset(VOCroot1, VOCjson_file1, size=0, enable_transforms=True, HW=[256, 256])

    HyperKroot_test = "/home/hostname/hostname-MTRRNetv2/data/EndoData"
    HyperKJson_test = "/home/hostname/hostname-MTRRNetv2/data/EndoData/test.json"
    HyperK_data_test = HyperKDataset(root=HyperKroot_test, json_path=HyperKJson_test, start=369, end=372, size=200, enable_transforms=True, unaligned_transforms=False, if_align=True, HW=[256,256], flag=None)


    test_data = ConcatDataset([test_data1, test_data2, test_data3, VOCdataset1, HyperK_data_test])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, drop_last=False, pin_memory=True)


    best_val_loss = float('inf')
    num_epochs = opts.epochs

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f"[Train] Epoch {epoch}") as pbar:
            for batch in pbar:
                inputs, labels, fns = batch['input'], batch['target_t'], batch['fn']
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                preds = model(inputs)

                mse = mse_fn(preds, labels)
                ssim_l = ssim_loss(preds, labels)
                loss = mse + ssim_l

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'ssiml': ssim_l.item(), 'mse': mse.item()})

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_inputs = []
        with torch.no_grad():
            print(f"Validating at epoch {epoch}")
            with tqdm(test_loader, desc=f"[Val] Epoch {epoch}") as pbar:
                for batch in pbar:
                    inputs, labels, fns = batch['input'], batch['target_t'], batch['fn']
                    inputs, labels = inputs.to(device), labels.to(device)

                    preds = model(inputs)

                    mse = mse_fn(preds, labels)
                    ssim_l = ssim_loss(preds, labels)
                    loss = mse + ssim_l
                    val_loss += loss.item()

                    all_inputs.append(inputs)
                    all_preds.append(preds)
                    all_labels.append(labels)

        val_loss /= len(test_loader)

        visualize(torch.cat(all_inputs)[:], torch.cat(all_preds)[:], torch.cat(all_labels)[:], epoch)

        os.makedirs('./cls/cls_models', exist_ok=True)
        torch.save(model.state_dict(), "./cls/cls_models/latest.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model at epoch {epoch} with loss {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join('./cls/cls_models',f"model_{epoch}.pth"))
        else:
            print(f"Epoch {epoch} did not improve. Best loss:{best_val_loss:.4f}  now: {val_loss:.4f}")


