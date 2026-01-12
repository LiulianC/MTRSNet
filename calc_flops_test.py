import torch
from thop import profile, clever_format
from thop.vision.basic_hooks import count_convNd, count_linear
from vmamba import selective_scan_flop_jit, flops_selective_scan_fn

from MTRRNet import MTRREngine

model = MTRREngine(device='cuda')
model.eval()

EXCLUDE_DECODERS = [
    'netG_T.token_decoder0',
    'netG_T.token_decoder1',
    'netG_T.token_decoder2',
]

class NoOpDecoder(torch.nn.Module):
    def forward(self, tokens_list, resident_tokens_list, x_in):
        B, _, H, W = x_in.shape
        return torch.zeros(B, 6, H, W, device=x_in.device, dtype=x_in.dtype)

def _replace_module_by_path(root, dotted_path: str, new_mod: torch.nn.Module):
    parts = dotted_path.split('.')
    cur = root
    for p in parts[:-1]:
        if not hasattr(cur, p):
            raise AttributeError(f"Path segment '{p}' not found while resolving '{dotted_path}'")
        cur = getattr(cur, p)
    last = parts[-1]
    if not hasattr(cur, last):
        raise AttributeError(f"Target attribute '{last}' not found on '{'.'.join(parts[:-1])}'")
    setattr(cur, last, new_mod)

for path in EXCLUDE_DECODERS:
    try:
        _replace_module_by_path(model, path, NoOpDecoder())
        print(f"[FLOPs] Excluded module by replacement: {path}")
    except AttributeError as e:
        print(f"[FLOPs] Skip exclude '{path}': {e}")

mamba_flops = 0
swin_flops = 0
mamba_params = 0
swin_params = 0

def count_vmamba(m, x, y):
    global mamba_flops, mamba_params
    
    print(f"[DEBUG] Mamba module called: {m.__class__.__name__}")
    print(f"[DEBUG] Input shape: {x[0].shape if isinstance(x, (list, tuple)) else x.shape}")
    
    if isinstance(x, (list, tuple)):
        inp = x[0]
    else:
        inp = x
    
    if inp.ndim == 4:  
        if inp.shape[1] >= 96:  
            B, C, H, W = inp.shape
            L = H * W
            D = C
        else:
            B, H, W, C = inp.shape
            L = H * W
            D = C
    elif inp.ndim == 3:  
        B, L, D = inp.shape
    else:
        B, L, D = inp.shape[0], inp.shape[1], inp.shape[2]
    
    N = 16  
    
    if hasattr(m, 'Mamba_num'):
        mamba_layers = m.Mamba_num
        mha_layers = m.Trans_num
        print(f"[DEBUG] Mamba2Blocks_Standard配置: Mamba层数={mamba_layers}, MHA层数={mha_layers}")
    else:
        n_layer = 1
        mamba_layers = 1
        mha_layers = 0
        print(f"[DEBUG] 警告: 未找到n_layer属性，使用默认单层")
    
    print(f"[DEBUG] Mamba dimensions - B:{B}, L:{L}, D:{D}, N:{N}")
    
    mamba_flops_total = 0
    try:
        single_mamba_flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
        mamba_flops_total = single_mamba_flops * mamba_layers
        print(f"[DEBUG] 单层Mamba FLOPs: {single_mamba_flops}")
        print(f"[DEBUG] {mamba_layers}层Mamba总FLOPs: {mamba_flops_total}")
    except Exception as e:
        print(f"[DEBUG] flops_selective_scan_fn失败: {e}")
        single_mamba_flops = 8 * B * L * D * N
        mamba_flops_total = single_mamba_flops * mamba_layers
        print(f"[DEBUG] 使用备选计算: 单层={single_mamba_flops}, 总={mamba_flops_total}")
    
    mha_flops_total = 0
    if mha_layers > 0:
        qkv_flops = 3 * B * L * D * D
        num_heads = 8
        head_dim = D // num_heads
        attn_flops = B * num_heads * L * head_dim * L
        proj_flops = B * L * D * D
        mlp_flops = 2 * B * L * D * (4 * D)  
        
        single_mha_flops = qkv_flops + attn_flops + proj_flops + mlp_flops
        mha_flops_total = single_mha_flops * mha_layers
        print(f"[DEBUG] 单层MHA FLOPs: {single_mha_flops}")
        print(f"[DEBUG] {mha_layers}层MHA总FLOPs: {mha_flops_total}")
    
    total_flops = mamba_flops_total + mha_flops_total
    print(f"[DEBUG] Mamba2Blocks_Standard总FLOPs: {total_flops}")

    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64, device=inp.device)
    else:
        m.total_ops = m.total_ops.to(inp.device)

    m.total_ops += torch.tensor([total_flops], dtype=torch.float64, device=inp.device)
    mamba_flops += total_flops  
    print(f"[DEBUG] Mamba FLOPs累计: {mamba_flops}\n")

def count_transformer_block(m, x, y):
    global swin_flops, swin_params
    
    print(f"[DEBUG] Swin module called: {m.__class__.__name__}")

    if isinstance(x, (list, tuple)):
        inp = x[0]
    else:
        inp = x

    print(f"[DEBUG] Swin input shape: {inp.shape}")

    if inp.ndim == 4:  
        B, C, H, W = inp.shape
        N = H * W
    elif inp.ndim == 3:  
        B, N, C = inp.shape
    else:
        B, N, C = inp.shape[0], inp.shape[1], inp.shape[2]

    print(f"[DEBUG] Swin dimensions - B:{B}, N:{N}, C:{C}")

    qkv_flops = 3 * B * N * C * C
    num_heads = 8
    head_dim = C // num_heads
    attn_flops = B * num_heads * N * head_dim * N
    proj_flops = B * N * C * C
    mlp_flops = 2 * B * N * C * (4 * C)
    
    flops = qkv_flops + attn_flops + proj_flops + mlp_flops

    print(f"[DEBUG] Swin FLOPs breakdown - QKV: {qkv_flops}, Attn: {attn_flops}, Proj: {proj_flops}, MLP: {mlp_flops}")
    print(f"[DEBUG] Swin total FLOPs: {flops}")

    if not hasattr(m, "total_ops"):
        m.total_ops = torch.zeros(1, dtype=torch.float64, device=inp.device)
    else:
        m.total_ops = m.total_ops.to(inp.device)

    m.total_ops += torch.tensor([flops], dtype=torch.float64, device=inp.device)
    swin_flops += flops
    print(f"[DEBUG] Swin FLOPs accumulated: {swin_flops}\n")

from MTRR_token_modules import VSSTokenMambaModule, SwinTokenBlock, Mamba2Blocks_Standard, SwinTransformerBlock

try:
    from vmamba import Mamba
    MAMBA_CLASSES = (VSSTokenMambaModule, Mamba2Blocks_Standard, Mamba)
except:
    MAMBA_CLASSES = (VSSTokenMambaModule, Mamba2Blocks_Standard)

custom_ops = {
    **{cls: count_vmamba for cls in MAMBA_CLASSES},
    SwinTransformerBlock: count_transformer_block,
    torch.nn.Conv2d: count_convNd,
    torch.nn.ConvTranspose2d: count_convNd,
    torch.nn.Linear: count_linear,
}

try:
    from timm.models.layers import WindowAttention
    custom_ops[WindowAttention] = count_transformer_block
except:
    pass

hooked_modules = []

def debug_hook(m, x, y):
    hooked_modules.append(m.__class__.__name__)
    return None

for name, module in model.named_modules():
    module.register_forward_hook(debug_hook)

dummy_input = torch.randn(1, 3, 256, 256).to('cuda')

with torch.no_grad():

    model.I = dummy_input
    model.forward()
    output = model.fake_Ts[3]

print("Forward 中被调用的模块类型:")
called_modules = set(hooked_modules)
print(called_modules)
print(f"Total unique modules: {len(called_modules)}")

mamba_related = {'Mamba', 'VSSTokenMambaModule', 'Mamba2Blocks_Standard'} & called_modules
swin_related = {'SwinTokenBlock', 'WindowAttention', 'SwinTransformerBlock'} & called_modules

print(f"Mamba related modules called: {mamba_related}")
print(f"Swin related modules called: {swin_related}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params: {total_params/1e6:.3f}M")


flops, params = profile(
    model, 
    inputs=(dummy_input,), 
    custom_ops=custom_ops,
    verbose=True  
)

mamba_params = 0
swin_params = 0
other_params = 0

for name, module in model.named_modules():
    module_params = sum(p.numel() for p in module.parameters())
    
    if isinstance(module, MAMBA_CLASSES):
        mamba_params += module_params
    elif isinstance(module, (SwinTokenBlock, SwinTransformerBlock)):
        swin_params += module_params
    elif 'WindowAttention' in str(type(module)):
        swin_params += module_params
    else:
        other_params += module_params

total_calculated_params = mamba_params + swin_params + other_params
total_actual_params = sum(p.numel() for p in model.parameters())

print(f"参数统计验证:")
print(f"Mamba参数: {mamba_params/1e6:.3f}M")
print(f"Swin参数: {swin_params/1e6:.3f}M") 
print(f"其他参数: {other_params/1e6:.3f}M")
print(f"计算总参数: {total_calculated_params/1e6:.3f}M")
print(f"实际总参数: {total_actual_params/1e6:.3f}M")

flops, params = clever_format([flops, params], "%.3f")

print(f"Total FLOPs: {flops}")
print(f"Total Params: {params}")
print(f"Mamba Params: {mamba_params/1e6:.2f} M, FLOPs: {mamba_flops/1e9:.2f} G")
print(f"Swin Params: {swin_params/1e6:.2f} M, FLOPs: {swin_flops/1e9:.2f} G")




if mamba_flops < 1e9:  
    print("警告: Mamba FLOPs可能统计不完整！")
    print("建议检查:")
    print("1. Mamba模块是否正确注册到custom_ops")
    print("2. flops_selective_scan_fn函数是否正确实现")
    print("3. 输入维度处理是否正确")














if mamba_flops < 1e9:  
    print("\n⚠️  Mamba FLOPs统计确实有问题!")
    print("可能原因:")
    print("1. flops_selective_scan_fn函数计算不准确")
    print("2. Mamba模块的输入维度可能比预期的小")
    print("3. 某些Mamba模块可能没有被正确hook")