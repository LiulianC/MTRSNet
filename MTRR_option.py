import argparse
from types import SimpleNamespace
import torch

def _need_weight_decay(param_name: str, p: torch.nn.Parameter) -> bool:
    """Predicate for whether a parameter should use weight decay (True/False)."""
    no_decay_flags = ['raw_gamma', 'norm', 'bn', 'running_mean', 'running_var']
    if p.dim() == 1 and 'weight' in param_name:
        return False
    if any(x in param_name.lower() for x in no_decay_flags):
        return False
    return True
 

def build_train_opts(argv=None):
    """
    Build training options (opts) for train.py.
    Mirrors the defaults currently used in train.py so behavior stays the same.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--model_dir', type=str, default='./model_fit', help='the model dir')
    parser.add_argument('--save_dir', type=str, default='./results', help='the results saving dir')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=57117)
    parser.add_argument('--throttle_ms', type=int, default=0, help='sleep milliseconds after each optimizer.step(); 0 to disable')

    opts = parser.parse_args(argv)

    opts.batch_size_train = 1
    opts.batch_size_test = 4

    opts.shuffle = True
    opts.display_id = -1 
    opts.num_workers = 0

    opts.always_print = 0
    opts.debug_monitor_layer_stats = 0
    opts.debug_monitor_layer_grad = 0

    opts.training = False
    
    opts.epoch = 121
    opts.es_patience = 20
    if opts.training:
        opts.sampler_size1 = 0
        opts.sampler_size2 = 0
        opts.sampler_size3 = 800
        opts.sampler_size4 = 0
        opts.sampler_size5 = 1200
        opts.test_size = [200, 0, 0, 0, 200, 200]
    else:
        opts.sampler_size1 = 0
        opts.sampler_size2 = 0
        opts.sampler_size3 = 1
        opts.sampler_size4 = 0
        opts.sampler_size5 = 1
        opts.test_size = [200, 0, 0, 0, 200, 200]

    opts.model_path = './model_118.pth'
    opts.reset_best = False
    opts.base_lr = 1e-4

    opts.color_enhance = False
    opts.AdditionSkip_en = True

    opts.scheduler_type = 'plateau'  
    
    if not hasattr(opts, 'base_lr'):
        opts.base_lr = 1.0e-4

    return opts



def get_lr_map(profile: str = 'train'):
    """
    Return the per-module learning rate map.
    profile: 'train' or 'debug'; currently both use decoders-only small LR as in the existing scripts.
    """
    if profile not in ('train', 'debug'):
        profile = 'train'

    if profile == 'train':
        return {
            'token_decoder3': 9.9e-05,
            'token_decoder2': 9.9e-05,
            'token_decoder1': 9.9e-05,
            'token_decoder0': 9.9e-05,
        }
    return {
        'token_decoder3': 1.0e-04,
        'token_decoder2': 1.0e-04,
        'token_decoder1': 1.0e-04,
        'token_decoder0': 1.0e-04,
    }


def make_param_groups(model: torch.nn.Module, lr_map: dict, base_lr: float, *, profile: str = 'train'):
    """
    Build param groups for optimizer from model parameters using lr_map and a default base_lr.
    Returns (param_groups, stats) where stats includes counts per group.
    Weight decay setup:
      - profile='train': wd_decay=0.0, wd_no_decay=0.0 (same as current train.py)
      - profile='debug': wd_decay=1e-4, wd_no_decay=0.0 (same as current debug_train.py)
    """
    if profile == 'debug':
        wd_decay, wd_no_decay = 1e-4, 0.0
    else:
        wd_decay, wd_no_decay = 0.0, 0.0

    module_params = {k: {'decay': [], 'no_decay': []} for k in lr_map.keys()}
    module_params['other'] = {'decay': [], 'no_decay': []}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        need_decay = _need_weight_decay(n, p)
        matched = False
        for module_name in lr_map.keys():
            if n.startswith(module_name):
                if need_decay:
                    module_params[module_name]['decay'].append(p)
                else:
                    module_params[module_name]['no_decay'].append(p)
                matched = True
                break
        if not matched:
            if need_decay:
                module_params['other']['decay'].append(p)
            else:
                module_params['other']['no_decay'].append(p)

    param_groups = []
    for module_name, lr in lr_map.items():
        if module_params[module_name]['decay']:
            param_groups.append({
                'params': module_params[module_name]['decay'],
                'lr': lr,
                'weight_decay': wd_decay,
                'name': f'{module_name}_decay'
            })
        if module_params[module_name]['no_decay']:
            param_groups.append({
                'params': module_params[module_name]['no_decay'],
                'lr': lr,
                'weight_decay': wd_no_decay,
                'name': f'{module_name}_no_decay'
            })

    if module_params['other']['decay']:
        param_groups.append({
            'params': module_params['other']['decay'],
            'lr': base_lr,
            'weight_decay': wd_decay,
            'name': 'other_decay'
        })
    if module_params['other']['no_decay']:
        param_groups.append({
            'params': module_params['other']['no_decay'],
            'lr': base_lr,
            'weight_decay': wd_no_decay,
            'name': 'other_no_decay'
        })

    stats = {g['name']: sum(p.numel() for p in g['params']) for g in param_groups}
    return param_groups, stats


def build_optimizer_and_scheduler(model: torch.nn.Module, opts, *, profile: str = 'train'):
    """
    Construct optimizer and scheduler using centralized policy.
    Returns (optimizer, scheduler, lr_map, param_groups_stats)
    """
    lr_map = get_lr_map(profile)
    base_lr = getattr(opts, 'base_lr', 1.0e-4)

    param_groups, stats = make_param_groups(model, lr_map, base_lr, profile=profile)

    optimizer = torch.optim.Adam(param_groups, betas=(0.5, 0.999), eps=1e-8)

    scheduler_type = getattr(opts, 'scheduler_type', 'plateau')
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.75,
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=1,
            min_lr=1e-8,
            eps=1e-8,
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7,
        )
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    return optimizer, scheduler, lr_map, stats


def build_early_stopping(opts):
    from early_stop import EarlyStopping
    patience = getattr(opts, 'es_patience', 20)
    delta = getattr(opts, 'es_delta', 1e-4)
    verbose = getattr(opts, 'es_verbose', True)
    return EarlyStopping(patience=patience, delta=delta, verbose=verbose)









