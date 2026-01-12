import os
import torch
from MTRRNet import MTRREngine
from customloss import CustomLoss
from dataset.new_dataset1 import DSRTestDataset, HyperKDataset
from torch.utils.data import ConcatDataset
import math
import warnings
from MTRR_option import get_lr_map, build_optimizer_and_scheduler, build_train_opts

warnings.filterwarnings('ignore')

opts = build_train_opts()



step = 0
max_steps = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs('./debug', exist_ok=True)

if os.path.exists('./debug/debug-state.log'):
    os.remove('./debug/debug-state.log')
if os.path.exists('./debug/debug-grad.log'):
    os.remove('./debug/debug-grad.log')

print("Initializing model...")
model = MTRREngine(opts, device)

if opts.model_path and os.path.exists(opts.model_path):
    print(f"Loading checkpoint from {opts.model_path}")
    checkpoint = torch.load(opts.model_path, map_location=device, weights_only=False)
    model.netG_T.load_state_dict({k.replace('netG_T.', ''): v for k, v in checkpoint['netG_T'].items()}, strict=True)
    print("Checkpoint loaded successfully")
else:
    print("No checkpoint found, using random initialization")

def monitor_all_layers_independently(model):
    """
    独立监控所有层的 forward 输出，包括展开 Mamba 变体的内部层
    不依赖于 MTRRNet.py 的 monitor_layer_stats()
    """
    hooks = []

    def make_hook(layer_name):
        def hook_fn(mod, inp, output):
            if isinstance(output, torch.Tensor):
                mean = output.mean().item()
                std = output.std().item()
                min_val = output.min().item()
                max_val = output.max().item()
                median = output.median().item()
                l2_norm = torch.norm(output).item()

                msg = (f"{layer_name:<100} | {mean:>12.6e} | {std:>12.6e} | {min_val:>12.6e} | "
                       f"{max_val:>12.6e} | {median:>12.6e} | {l2_norm:>12.6e} | {tuple(output.shape)}")
                with open('./debug/debug-state.log', 'a') as f:
                    f.write(msg + '\n')
        return hook_fn

    try:
        from mamba_ssm.modules.mamba2 import Mamba2
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.mamba2_simple import Mamba2Simple
        has_mamba = True
    except ImportError:
        has_mamba = False
        Mamba = Mamba2 = Mamba2Simple = type(None)

    mamba_count = 0
    mamba2_count = 0
    mamba2simple_count = 0
    total_layers = 0

    from torch import nn
    for name, module in model.netG_T.named_modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue

        is_mamba_variant = False
        if has_mamba and isinstance(module, (Mamba, Mamba2, Mamba2Simple)):
            is_mamba_variant = True
            if isinstance(module, Mamba2):
                mamba2_count += 1
                module_type = "Mamba2"
            elif isinstance(module, Mamba2Simple):
                mamba2simple_count += 1
                module_type = "Mamba2Simple"
            elif isinstance(module, Mamba):
                mamba_count += 1
                module_type = "Mamba"

            hook = module.register_forward_hook(make_hook(f"{name} ({module_type})"))
            hooks.append(hook)
            total_layers += 1

            for sub_name, sub_module in module.named_modules():
                if sub_name:  
                    full_name = f"{name} ({module_type}).{sub_name}"
                    hook = sub_module.register_forward_hook(make_hook(full_name))
                    hooks.append(hook)
                    total_layers += 1
        else:
            if name:  
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
                total_layers += 1

    mamba_total = mamba_count + mamba2_count + mamba2simple_count
    print(f"Registered {total_layers} hooks for forward monitoring:")
    print(f"  - Total Mamba variants: {mamba_total}")
    print(f"    - Mamba: {mamba_count}")
    print(f"    - Mamba2: {mamba2_count}")
    print(f"    - Mamba2Simple: {mamba2simple_count}")
    return hooks


print("Registering independent forward hooks for all layers (including Mamba internals)...")
state_hooks = monitor_all_layers_independently(model)

print("Loading dataset...")
tissue_dir = '/home/hostname/hostname-MTRRVideo/data/tissue_real'
tissue_data = DSRTestDataset(
    datadir=tissue_dir,
    fns='/home/hostname/hostname-MTRRVideo/data/tissue_real_index/train1.txt',
    size=800,  
    enable_transforms=True,
    unaligned_transforms=False,
    if_align=True,
    real=True,
    HW=[256, 256],
)

HyperKroot = "/home/hostname/hostname-MTRRNetv2/data/EndoData"
HyperKJson = "/home/hostname/hostname-MTRRNetv2/data/EndoData/test.json"
HyperK_data = HyperKDataset(
    root=HyperKroot,
    json_path=HyperKJson,
    start=343,
    end=369,
    size=1200,  
    enable_transforms=True,
    unaligned_transforms=False,
    if_align=True,
    HW=[256, 256],
    flag=None,
    color_jitter=True
)

train_data = ConcatDataset([tissue_data, HyperK_data])
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opts.batch_size_train,
    shuffle=opts.shuffle,
    num_workers=opts.num_workers,
    drop_last=False,
    pin_memory=True
)

print(f"Dataset loaded: {len(train_data)} samples")

loss_function = CustomLoss().to(device)







LearnRate = opts.base_lr
lr_map = get_lr_map('Train')









optimizer, scheduler, lr_map, group_stats = build_optimizer_and_scheduler(model.netG_T, opts, profile='debug')

if opts.model_path and os.path.exists(opts.model_path):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded successfully")




print("\n" + "="*80)
print("Learning Rate Configuration:")
print("="*80)
for group in optimizer.param_groups:
    n_params = sum(p.numel() for p in group['params'])
    print(f"{group['name']:<40} LR: {group['lr']:.2e}  Params: {n_params:>10,}  WD: {group['weight_decay']}")
print("="*80 + "\n")

model.netG_T.train()

print("\n" + "="*120)
print("Starting debug training for 3 steps...")
print("="*120 + "\n")



for batch_idx, data in enumerate(train_loader):
    if step >= max_steps:
        break

    step += 1
    print(f"\n{'='*120}")
    print(f"DEBUG STEP {step}/{max_steps}")
    print(f"{'='*120}")

    with open('./debug/debug-state.log', 'a') as f:
        f.write(f"\n{'='*280}\n")
        f.write(f"STEP {step} - Forward Pass Statistics (Including Mamba Internal Layers)\n")
        f.write(f"{'='*280}\n")
        f.write(f"{'Layer Name':<100} | {'Mean':>12} | {'Std':>12} | {'Min':>12} | {'Max':>12} | {'Median':>12} | {'L2Norm':>12} | {'Shape'}\n")
        f.write(f"{'-'*280}\n")

    model.set_input(data)

    print("Running forward pass...")
    model.inference()

    visuals = model.get_current_visuals()
    train_input = visuals['I']
    train_ipt = visuals['Ic']
    train_label1 = visuals['T']
    train_label2 = visuals['R']
    train_fake_Ts = visuals['fake_Ts']
    train_fake_Rs = visuals['fake_Rs']
    train_rcmaps = visuals['c_map']

    print("Computing loss...")
    _, _, _, _, _, all_loss0 = loss_function(
        train_fake_Ts[0], train_label1, train_ipt, train_rcmaps, train_fake_Rs[0], train_label2
    )
    _, _, _, _, _, all_loss1 = loss_function(
        train_fake_Ts[1], train_label1, train_ipt, train_rcmaps, train_fake_Rs[1], train_label2
    )
    _, _, _, _, _, all_loss2 = loss_function(
        train_fake_Ts[2], train_label1, train_ipt, train_rcmaps, train_fake_Rs[2], train_label2
    )
    loss_table, mse_loss, vgg_loss, ssim_loss, fake_Ts_range_penalty, all_loss3 = loss_function(
        train_fake_Ts[3], train_label1, train_ipt, train_rcmaps, train_fake_Rs[3], train_label2
    )
    all_loss = 0.5*all_loss0 + 0.5*all_loss1 + 0.5*all_loss2 + 1.0*all_loss3

    print(f"Loss: {all_loss.item():.6f} | MSE: {mse_loss.item():.6f} | VGG: {vgg_loss.item():.6f} | SSIM: {ssim_loss.item():.6f}")

    print("Running backward pass...")
    optimizer.zero_grad()
    all_loss.backward()


    with open('./debug/debug-grad.log', 'a') as f:
        f.write(f"\n{'='*220}\n")
        f.write(f"STEP {step} - Gradient Statistics (Including Mamba Internal Parameters)\n")
        f.write(f"{'='*220}\n")
        f.write(f"{'Parameter Name':<100} | {'Grad Mean':>15} | {'Grad Std':>15} | {'Grad Min':>15} | {'Grad Max':>15} | {'Grad Norm':>15}\n")
        f.write(f"{'-'*220}\n")

    print("Collecting gradient statistics...")
    with open('./debug/debug-grad.log', 'a') as f:
        for name, param in model.netG_T.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_min = param.grad.min().item()
                grad_max = param.grad.max().item()
                grad_norm = torch.norm(param.grad).item()

                is_nan = math.isnan(grad_mean) or math.isnan(grad_std)
                is_inf = math.isinf(grad_mean) or math.isinf(grad_std)

                status = ""
                if is_nan:
                    status = " [NaN DETECTED!]"
                elif is_inf:
                    status = " [Inf DETECTED!]"
                elif abs(grad_norm) > 100:
                    status = " [Large Gradient]"
                elif abs(grad_norm) < 1e-6:
                    status = " [Vanishing]"

                msg = (f"{name:<100} | {grad_mean:>15.8e} | {grad_std:>15.8e} | "f"{grad_min:>15.8e} | {grad_max:>15.8e} | {grad_norm:>15.8e}{status}")
                f.write(msg + '\n')                
                    



    optimizer.step()

    print(f"Step {step} completed.")

print("\n" + "="*120)
print("Debug training completed!")
print("="*120)
print(f"\nResults saved to:")
print(f"  - Forward pass statistics (with Mamba internals): ./debug/debug-state.log")
print(f"  - Gradient statistics (with Mamba internals): ./debug/debug-grad.log")
print("\nYou can review the logs to analyze layer outputs and gradients.")
print("\nNote: Mamba internal layers (in_proj, conv1d, act, norm, out_proj, x_proj, dt_proj) are expanded in-place.")
print("If some Mamba layers are not captured, it's because Mamba uses fused CUDA kernels.")
print("Consider setting use_mem_eff_path=False in Mamba initialization for full monitoring.")
