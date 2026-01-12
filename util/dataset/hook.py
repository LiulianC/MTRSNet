import torch
import torch.nn as nn

def monitor_layer_stats(model, input_data):
    """为模型每一层注册前向钩子，打印输出张量的均值和标准差"""
    hooks = []
    
    def hook_fn(module, input, output, layer_name):
        if isinstance(output, torch.Tensor):
            mean = output.mean().item()
            std = output.std().item()
            print(f"Layer: {layer_name:20} | Mean: {mean:8.4f} | Std: {std:8.4f} | Shape: {tuple(output.shape)}")
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):  
            hook = module.register_forward_hook(
                lambda m, inp, out, name=name: hook_fn(m, inp, out, name)
            )
            hooks.append(hook)
    
    with torch.no_grad():
        model(input_data)
    
    for hook in hooks:
        hook.remove()