                                 
                             
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.vision_transformer import PatchEmbed
from timm.models.swin_transformer import SwinTransformerBlock,SwinTransformerStage
from timm.layers import LayerNorm2d
import math
from timm.models.layers import DropPath
from typing import List


def init_all_weights(model: nn.Module):
\
\
\
\
\
\
\
\
       
    gelu_gain = nn.init.calculate_gain('relu')

    def _init(m):
                                                                                
        if isinstance(m, Mamba2):
            return        
        if isinstance(m, Mamba2Simple):
            return        
        if isinstance(m, Mamba):
            return        
        if isinstance(m, VSSTokenMambaModule):
            return        
        if isinstance(m, Mamba2Blocks_Standard):
            return        
        
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=gelu_gain)
            if m.bias is not None:
                                        
                nn.init.uniform_(m.bias, -0.1, 0.1)

        elif isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.xavier_uniform_(m.weight, gain=gelu_gain)
            if m.bias is not None:
                                        
                nn.init.uniform_(m.bias, -0.1, 0.1)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
            if getattr(m, 'weight', None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                                        
                nn.init.uniform_(m.bias, -0.1, 0.1)
                
        elif isinstance(m, nn.PReLU):
            with torch.no_grad():
                m.weight.fill_(0.08)

                          
        elif isinstance(m, TokenPatchEmbed):
            nn.init.normal_(m.weight, std=0.02)

    model.apply(_init)


class AAF(nn.Module):
\
\
\
       
    def __init__(self, in_channels: int, num_inputs: int):
        super(AAF, self).__init__()
        self.in_channels = in_channels
        self.num_inputs = num_inputs

    @torch.no_grad()
    def _check(self, features: List[torch.Tensor]):
        assert isinstance(features, (list, tuple)) and len(features) == self.num_inputs, \
            f"Expect {self.num_inputs} inputs, got {len(features)}."
        shapes = [tuple(x.shape) for x in features]
        assert all(s == shapes[0] for s in shapes), \
            f"All inputs must share the same shape, got {shapes}."
        x = features[0]
        assert x.dim() in (3, 4), \
            f"Each input must be 3D [B,L,C] or 4D [B,C,H,W], got dim={x.dim()}."
        if x.dim() == 4:
            B, C, H, W = x.shape
            assert C == self.in_channels, \
                f"in_channels={self.in_channels} but got C={C}."
        else:                
            B, L, C = x.shape
            assert C == self.in_channels, \
                f"in_channels={self.in_channels} but got C={C}."

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
\
\
\
\
\
           
        self._check(features)

        x0 = features[0]
        if x0.dim() == 4:
                                                             
            x = torch.stack(features, dim=1)
                                                     
            weights = torch.softmax(x, dim=1)                            
            out = (weights * x).sum(dim=1)                            
                                                             
            return out
        else:
                                                       
            x = torch.stack(features, dim=1)
            weights = torch.softmax(x, dim=1)                         
            out = (weights * x).sum(dim=1)                         
            return out


class TokenPatchEmbed(nn.Module):
                                                            
    def __init__(self, img_size, patch_size, in_chans, embed_dim, use_sincos_pos=True, pos_init_scale=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.use_sincos_pos = use_sincos_pos

                 
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

                                   
        if self.use_sincos_pos:
            self.pos_alpha = nn.Parameter(torch.tensor(float(pos_init_scale)))
                                              
            self.pos_alpha.register_hook(self._pos_alpha_grad_hook)

                                                        
            pos = self._build_2d_sincos_pos_embed(self.grid_size, self.grid_size, embed_dim, device='cpu')
            self.register_buffer('pos_embed', pos, persistent=False)                    
        else:
            self.register_buffer('pos_embed', None, persistent=False)

    def _pos_alpha_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
                                        
        gmax = 0.1
        return grad.clamp(min=-gmax, max=gmax)

    def forward(self, x):
                                                         
        x = self.proj(x)
        B, C, Ht, Wt = x.shape
                                                                       
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

                                      
        if self.use_sincos_pos:
            if (Ht == self.grid_size) and (Wt == self.grid_size) and (self.pos_embed is not None):
                x = x + self.pos_alpha * self.pos_embed                 
            else:
                                      
                pos_dyn = self._build_2d_sincos_pos_embed(Ht, Wt, self.embed_dim, device=x.device)                 
                x = x + self.pos_alpha * pos_dyn

        return x                               

    @staticmethod
    def _get_1d_sincos_pos_embed(embed_dim, length, device):
        if embed_dim % 2 != 0:
            extra = 1
            d = embed_dim - 1
        else:
            extra = 0
            d = embed_dim

        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)         
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))          

        sinusoid = position * div_term            
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=1)          

        if extra == 1:
            pad = torch.zeros(length, 1, device=device, dtype=torch.float32)
            emb = torch.cat([emb, pad], dim=1)

        return emb

    @classmethod
    def _build_2d_sincos_pos_embed(cls, H, W, embed_dim, device='cpu'):
        dim_h = embed_dim // 2
        dim_w = embed_dim - dim_h

        pos_h = cls._get_1d_sincos_pos_embed(dim_h, H, device)              
        pos_w = cls._get_1d_sincos_pos_embed(dim_w, W, device)              

        pos_h_broadcast = pos_h[:, None, :].repeat(1, W, 1)                 
        pos_w_broadcast = pos_w[None, :, :].repeat(H, 1, 1)                 
        pos_2d = torch.cat([pos_h_broadcast, pos_w_broadcast], dim=2)                     

        pos_2d = pos_2d.view(1, H * W, embed_dim)
        return pos_2d


from typing import Optional

class OverlapTokenPatchEmbed(nn.Module):
                                                               

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 6,
        in_chans: int = 3,
        embed_dim: int = 96,
        use_sincos_pos: bool = True,
        pos_init_scale: float = 0.1,
        stride: Optional[int] = None,
        padding: int = 0,
        default_grid_size: int = 64,                       
    ):
\
\
\
\
\
\
           
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.padding = padding
        self.embed_dim = embed_dim
        self.use_sincos_pos = use_sincos_pos
        self.default_grid_size = default_grid_size

        self.num_patches: Optional[int] = None

                                                                 
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.use_sincos_pos:
            self.pos_alpha = nn.Parameter(torch.tensor(float(pos_init_scale)))
            self.pos_alpha.register_hook(self._pos_alpha_grad_hook)
        else:
            self.pos_alpha = None

        nn.init.normal_(self.mask_token, std=0.02)

    def _pos_alpha_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        gmax = 0.1
        return grad.clamp(min=-gmax, max=gmax)

    def _build_random_token_mask(self, batch_size: int, num_tokens: int, mask_ratio: float, device) -> torch.Tensor:
        num_mask = int(round(num_tokens * float(mask_ratio)))
        num_mask = max(0, min(num_tokens, num_mask))
        mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
        if num_mask == 0:
            return mask
        noise = torch.rand(batch_size, num_tokens, device=device)
        ids = torch.argsort(noise, dim=1)[:, :num_mask]
        mask.scatter_(1, ids, True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.0,
        return_mask: bool = False,
    ):
\
\
\
           
        x = self.proj(x)                                       
        B, C, Ht, Wt = x.shape
        self.num_patches = Ht * Wt

                                          
                                                                                 
                                                                                               

        x = x.flatten(2).transpose(1, 2)               
        x = self.norm(x)

        if token_mask is None and mask_ratio > 0:
            token_mask = self._build_random_token_mask(B, Ht * Wt, mask_ratio, x.device)
        if token_mask is not None:
            token_mask = token_mask.to(device=x.device, dtype=torch.bool)
            if token_mask.shape != (B, Ht * Wt):
                raise ValueError(f"token_mask shape must be {(B, Ht * Wt)}, got {tuple(token_mask.shape)}")
            mask_token = self.mask_token.to(dtype=x.dtype).expand(B, Ht * Wt, -1)
            x = torch.where(token_mask.unsqueeze(-1), mask_token, x)

                            
        if self.use_sincos_pos and self.pos_alpha is not None:
            pos = self._build_2d_sincos_pos_embed(
                Ht, Wt, self.embed_dim, device=x.device
            ).to(dtype=x.dtype)                 
            x = x + self.pos_alpha * pos

        if return_mask:
            if token_mask is None:
                token_mask = torch.zeros(B, Ht * Wt, dtype=torch.bool, device=x.device)
            return x, token_mask, (Ht, Wt)
        return x                     

    @staticmethod
    def _get_1d_sincos_pos_embed(embed_dim, length, device):
        if embed_dim % 2 != 0:
            extra = 1
            d = embed_dim - 1
        else:
            extra = 0
            d = embed_dim

        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d)
        )

        sinusoid = position * div_term
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=1)

        if extra == 1:
            pad = torch.zeros(length, 1, device=device, dtype=torch.float32)
            emb = torch.cat([emb, pad], dim=1)

        return emb

    @classmethod
    def _build_2d_sincos_pos_embed(cls, H, W, embed_dim, device="cpu"):
        dim_h = embed_dim // 2
        dim_w = embed_dim - dim_h

        pos_h = cls._get_1d_sincos_pos_embed(dim_h, H, device)              
        pos_w = cls._get_1d_sincos_pos_embed(dim_w, W, device)              

        pos_h_broadcast = pos_h[:, None, :].repeat(1, W, 1)                    
        pos_w_broadcast = pos_w[None, :, :].repeat(H, 1, 1)                    
        pos_2d = torch.cat([pos_h_broadcast, pos_w_broadcast], dim=2)             

        pos_2d = pos_2d.view(1, H * W, embed_dim)
        return pos_2d

                                                                                                
                                                  

                                                   
                                    
                   
                   
                 
                    
               
                
                           
   

                                                    
                                    
                   
                    
                 
                    
               
                
                           
   

                                                     
                                    
                   
                    
                 
                    
                
                
                           
   


class MambaTokenBlock(nn.Module):
                                               
    def __init__(self, dim, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),
                Mamba(dim)
            ))
    
    def forward(self, x):
                      
        for block in self.blocks:
            x = x + block(x)                          
        return x

from vmamba import VSSBlock,SS2D
from collections import OrderedDict
class VSSTokenMambaModule(nn.Module):
    def __init__(
        self,  
        depths=[9], 
        dims=[192], 
                                   
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2", 
                                   
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
                                   
        drop_path_rate=0.1, 
        use_checkpoint=False,  
                                   
        posembed=False,
        _SS2D=SS2D,
        channel_first=False,
                                   
        **kwargs,        
    ):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]                     
        self.channel_first = channel_first
        self.dims = dims
        self.num_layers = 1
        

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            self.layers.append(self._make_layer(                                                                     
                dim = self.dims[0],
                depth = depths[0],
                drop_path = dpr[1],
                                  
                use_checkpoint=use_checkpoint,
                channel_first=self.channel_first,
                                   
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                                   
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                                   
                _SS2D=_SS2D,
            ))

    def forward(self, x):
                       
                                                      
                           
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x  
    
    @staticmethod
    def _make_layer(
        dim=96, 
        depth=9,
        drop_path=0.1, 
        use_checkpoint=False, 
        downsample=nn.Identity(),
        channel_first=False,
                                     
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.05, 
        ssm_init="v0",
        forward_type="v2",
                                     
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
                                     
        **kwargs,
    ):
                                                                       
        depth = depth
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))    

from mamba_ssm.modules.mamba2 import Mamba2                 
from mamba_ssm.modules.mamba_simple import Mamba                 
from mamba_ssm.modules.mamba2_simple import Mamba2Simple                 
class Mamba2Blocks(nn.Module):
    def __init__(self, dim, num_blocks=1, drop_path_rate=0.05, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                Mamba2(d_model=dim,d_state=64,d_conv=4,expand=2),
                                                                                    
                                                                  
                nn.Dropout(drop_path_rate),
            ))
    def forward(self, x):
                         

        if self.channel_first:                 
            x = x.permute(0,2,3,1).contiguous()                

        B,H,W,C = x.shape
        
                  
        x = x.view(B, -1, C).contiguous()             

        for block in self.blocks:
            x = block(x)                          

        x = x.view(B, H, W, C).contiguous()                

        if self.channel_first:     
            x = x.permute(0,3,1,2).contiguous()                

        return x           


from functools import partial
from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Mamba2Blocks_Standard(nn.Module):
    def __init__(
        self,
        d_model: int,         
        n_layer: int,          
        d_intermediate: int,               

                          
        ssm_cfg={
            "layer": "Mamba1",                
                            
            "d_state": 16,                
            "d_conv": 4,                 
            "expand": 2,                
        },
                                    
                                         
                                        

                           
                   
                                                
                                 
                                      
                          
                          

                          
                                
                                         
                                       
                             
                             
                                    
                                         
                                    
                            
                                
            
                                      
                                         
                                        

                                       
        attn_layer_idx=None,                          
        attn_cfg={
            "num_heads": 8,                  
                                                
            "num_heads_kv": None,                               
                                                                      
            "head_dim": None,                                         
            "mlp_dim": 0,                            
            "qkv_proj_bias": True,             
            "out_proj_bias": True,            
            "softmax_scale": None,                 
            "causal": False,                     
            "d_conv": 0,                           
            "rotary_emb_dim": 0,                
            "rotary_emb_base": 10000.0,           
            "rotary_emb_interleaved": False,             
        },
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=False,
                                                      
        residual_in_fp32=True,
        device='cuda',
                                                                     
        dtype=None,
        channel_first=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.channel_first = channel_first
        self.n_layer = n_layer 

        self.Mamba_num = n_layer
        if attn_cfg is not None:
            self.Mamba_num = self.n_layer/2
        self.Trans_num = self.n_layer - self.Mamba_num


                                                         
                                                    
                                                                                                 
                                                                                     
                                                                       

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

                                                  
        if attn_layer_idx is None:
            attn_layer_idx = [i for i in range(1, n_layer, 2)]            
        
                           
        if attn_cfg is None:
            attn_cfg = {}
        
                                               
                                                          
                                                  
        
                                      
        _default_ssm = {
            "layer": "Mamba1",            
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "bias": False,
            "conv_bias": True,
        }
        _ssm_cfg_merged = dict(_default_ssm)
        if ssm_cfg is not None:
            _ssm_cfg_merged.update(ssm_cfg)

                    
                                             

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=_ssm_cfg_merged,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,                    
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        
                
                                                                
                                                                         

                                                                                                 
                                                                                                    
                                                                                        
                                                
                                                                   
                                                                     
                                                   
                   
                                                                                                            
                                             
                   
               
               
                                      


        self.layer_scales = None

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,                    
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x, inference_params=None, **mixer_kwargs):

                         

        if self.channel_first:                 
            x = x.permute(0,2,3,1).contiguous()                

        B,H,W,C = x.shape
        
                  
        hidden_states = x.view(B, -1, C).contiguous()             

        residual = None
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
            if self.layer_scales is not None:
                scale_param = self.layer_scales[idx]
                clamp_max = self.layer_scale_max if self.layer_scale_max is not None else None
                if clamp_max is not None:
                    scale_param = torch.clamp(scale_param, min=0.0, max=clamp_max)
                scale = scale_param.to(hidden_states.dtype).view(1, 1, -1)
                hidden_states = hidden_states * scale
                if residual is not None:
                    residual = residual * scale
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
                                                                     
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        hidden_states = hidden_states.view(B, H, W, C).contiguous()                

        if self.channel_first:     
            hidden_states = hidden_states.permute(0,3,1,2).contiguous()                    

        return hidden_states


class SwinTokenBlock(nn.Module):
                                                   
    def __init__(self, dim, input_resolution, num_heads, window_size, num_blocks=1):
        super().__init__()
        self.input_resolution = input_resolution

        blocks = []
        for i in range(num_blocks):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=4.0,
                    attn_drop=0.05,
                    drop_path=0.05,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
                                               
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, C).contiguous()

        for block in self.blocks:
            x = block(x)                             

        x = x.view(B, N, C)
        return x


from MTRR_RD_modules import FrequencyProcessor,ChannelAttention

class EncoderUnit(nn.Module):
                                               
    def __init__(self, ori_img_size=256, embed_dim=96, mamba_blocks=2, swin_blocks=2, grid_size=64, window_size=8, drop_branch_prob=0.1, 
                 need_downsample=False, need_freqAttention=False, need_channelAttention=False):
        super().__init__()
        self.img_size = ori_img_size
        self.grid_size = grid_size
        self.drop_branch_prob = drop_branch_prob
                                                                                                
        self.need_downsample = need_downsample
        self.need_freqAttention = need_freqAttention
        self.need_channelAttention = need_channelAttention

        if self.need_freqAttention:
            self.freqatt = FrequencyProcessor(channels=embed_dim, int_size=2*embed_dim)
        if self.need_channelAttention:
            self.channelatt = ChannelAttention(dim=embed_dim, num_heads=2, bias=True)


                                           
        self.ghost_grad_coeff = 0.02

                               
        if need_downsample is True:
                                              
            self.downSample = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=2, stride=2, padding=0, bias=False)        

                            
                                                                                                                                                                     
                                              
                                                                                                                                        
        self.mamba_processor = Mamba2Blocks_Standard(d_model=embed_dim, n_layer=mamba_blocks, d_intermediate=2*embed_dim) if mamba_blocks > 0 else None

        
        if swin_blocks > 0:
            input_resolution = (self.grid_size, self.grid_size) 
            num_heads = max(1, embed_dim // 32)
            self.swin_processor = SwinTokenBlock(embed_dim, input_resolution, num_heads, window_size, swin_blocks)
                                                 
        else:
            self.swin_processor = None

              
        if self.mamba_processor is not None and self.swin_processor is not None:
                                             
            self.fusion = nn.Conv2d(in_channels=embed_dim*2,out_channels=embed_dim,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
            self.fusion_out = nn.Identity()
        self.out = nn.Identity()
        

    def forward(self, x):
                      

        if self.need_downsample is True:
            B, N, C = x.shape
            x = x.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))
            x = self.downSample(x)                     
            B, C, H, W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(B, H*W, C)

        if self.need_freqAttention:              
            B, N, C = x.shape
            x = x.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))            
            x = self.freqatt(x)
            B, C, H, W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(B, H*W, C)            

        if self.need_channelAttention:              
            B, N, C = x.shape
            x = x.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))            
            x = self.channelatt(x)
            B, C, H, W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(B, H*W, C)            

        low_tokens = x.contiguous()             
        high_tokens = x.contiguous()            
        B,N,C = x.shape

              
        if self.mamba_processor is not None:
            low_tokens = low_tokens.view(B, int(N**0.5), int(N**0.5), C).contiguous()
            low_tokens = self.mamba_processor(low_tokens)
            low_tokens = low_tokens.view(B, N, C).contiguous()
        else:
            low_tokens = torch.zeros_like(low_tokens)

        if self.swin_processor is not None:
            high_tokens = self.swin_processor(high_tokens)
        else:
            high_tokens = torch.zeros_like(high_tokens)

                                   
                              
                                   
        if self.training and self.mamba_processor is not None and self.swin_processor is not None and getattr(self, 'drop_branch_prob', 0.0) > 0:
                                
            keep_scale = 1.0 / (1.0 - self.drop_branch_prob)
            rand_val = torch.rand(1, device=x.device)

            if rand_val < self.drop_branch_prob:
                                                             
                                                     
                ghost = self.ghost_grad_coeff * (low_tokens - low_tokens.detach())
                return keep_scale * high_tokens + ghost

            elif (rand_val > self.drop_branch_prob) and (rand_val < 2 * self.drop_branch_prob):
                                                            
                ghost = self.ghost_grad_coeff * (high_tokens - high_tokens.detach())
                return keep_scale * low_tokens + ghost

                                   
                             
                                   
        if self.mamba_processor is not None and self.swin_processor is not None:
            B, N, C = low_tokens.shape
            low_tokens = low_tokens.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))            
            high_tokens = high_tokens.permute(0,2,1).contiguous().view(B, C, int(N**0.5), int(N**0.5))    

            fused_tokens = self.fusion(torch.cat([low_tokens, high_tokens],dim=1))
            fused_tokens = self.fusion_out(fused_tokens)

            B, C, H, W = low_tokens.shape
            fused_tokens = fused_tokens.permute(0,2,3,1).contiguous().view(B, H*W, C)   


                                                     
        elif self.mamba_processor is not None:
            fused_tokens = low_tokens
        else:
            fused_tokens = high_tokens

        fused_tokens = self.out(fused_tokens)

        return fused_tokens


class Encoder(nn.Module):
                                     
    def __init__(self, in_chans=3, embed_dim=96, mamba_blocks=[2, 2, 2, 2], swin_blocks=[2, 2, 2, 2], drop_branch_prob=0.2):
        super().__init__()
        
                                                                                                               


                                                           
        self.patchembed = OverlapTokenPatchEmbed(
                img_size=256,
                patch_size=6,
                in_chans=in_chans,
                embed_dim=embed_dim,
                stride=4,
                padding=1,
                default_grid_size=64,
            )

        self.encoder_unit0 = EncoderUnit(embed_dim=96, grid_size=64, ori_img_size=256, mamba_blocks=mamba_blocks[0], swin_blocks=swin_blocks[0], 
                                    window_size=8, drop_branch_prob=drop_branch_prob, need_downsample=False, need_freqAttention=True)
        
        self.encoder_unit1 = EncoderUnit(embed_dim=192, grid_size=32, ori_img_size=256, mamba_blocks=mamba_blocks[1], swin_blocks=swin_blocks[1], 
                                    window_size=8, drop_branch_prob=drop_branch_prob, need_downsample=True, need_channelAttention=True)

                                                                                                       
                                                                                                        

        self.encoder_unit2 = EncoderUnit(embed_dim=384, grid_size=16, ori_img_size=256, mamba_blocks=mamba_blocks[2], swin_blocks=swin_blocks[2], 
                                    window_size=4, drop_branch_prob=drop_branch_prob, need_downsample=True, need_channelAttention=True)
        
                                                                                                                                                   
                                                                                                                                         
    
    def forward(self, x_in, token_mask: Optional[torch.Tensor] = None, mask_ratio: float = 0.0, return_token_mask: bool = False):
                                
        tokens_list = []
        
        if return_token_mask:
            x_emb1, token_mask, token_grid = self.patchembed(
                x_in,
                token_mask=token_mask,
                mask_ratio=mask_ratio,
                return_mask=True,
            )
        else:
            x_emb1 = self.patchembed(
                x_in,
                token_mask=token_mask,
                mask_ratio=mask_ratio,
                return_mask=False,
            )                          
        tokens_list.append(x_emb1)
        
        tokens = self.encoder_unit0(x_emb1)                           
        tokens_list.append(tokens)

        tokens = self.encoder_unit1(tokens)                            
        tokens_list.append(tokens)

        
        tokens = self.encoder_unit2(tokens)                            
        tokens_list.append(tokens)

                                                                       
                                    
            
        if return_token_mask:
            return tokens_list, token_mask, token_grid
        return tokens_list

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode, 
            align_corners=self.align_corners
        )

class SubNet(nn.Module):
                                   
    def __init__(self, embed_dims=[96,192,384,768], mam_blocks=[6,6,6,6], use_rev=False):
        super().__init__()
        self.embed_dims = embed_dims
                         
        self.use_rev = bool(use_rev)

        self.upsample1 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dims[1], embed_dims[1]//2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[1]//2, affine=True),
            nn.GELU(),
        )
        self.upsample2 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dims[2], embed_dims[2]//2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[2]//2, affine=True),
            nn.GELU(),
        )
                                         
                                                                                
                                                                                                                                 
                                                               
                        
           
        
        self.downsample0 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0]*2, kernel_size=2, stride=2, bias=False),        
            nn.InstanceNorm2d(embed_dims[0]*2, affine=True),
            nn.GELU(),
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[1]*2, kernel_size=2, stride=2, bias=False),        
            nn.InstanceNorm2d(embed_dims[1]*2, affine=True),
            nn.GELU(),
        )
                                           
                                                                                                     
                                                              
                        
           
        
                  
        self.mamba_blocks = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.mamba_blocks.append(nn.Sequential(
                                                                                                                             
                                                                                                                     
                Mamba2Blocks_Standard(d_model=embed_dims[i], n_layer=mam_blocks[i], d_intermediate=2*embed_dims[i], channel_first=True),
                ChannelAttention(dim=embed_dims[i], num_heads=2, bias=False)
            ))

        

        alpha_init_value = 0.7           
        channels = embed_dims
        self.alpha0 = nn.Parameter(alpha_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha1 = nn.Parameter(alpha_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha2 = nn.Parameter(alpha_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        
                                                                                           
                                                                                                      
    
        self.deconv_o0 = nn.Conv2d(in_channels=embed_dims[1],out_channels=embed_dims[0],kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.deconv_o0_f0 = nn.Conv2d(in_channels=embed_dims[1],out_channels=embed_dims[0],kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        
        self.deconv_o1 = nn.Conv2d(in_channels=embed_dims[2],out_channels=embed_dims[1],kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.deconv_o1_f1 = nn.Conv2d(in_channels=embed_dims[2],out_channels=embed_dims[1],kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        
                                                                                                                                                  
        self.deconv_o2_f2 = nn.Conv2d(in_channels=embed_dims[3],out_channels=embed_dims[2],kernel_size=3,stride=1,padding=1,padding_mode='reflect')

    def forward(self, tokens_list, use_eval=True):
                                                                               

        return self._forward_noreverse(tokens_list)

    def _forward_noreverse(self, tokens_list):
                                                          

        if tokens_list[0].ndim == 3:        
            for i, tokens in enumerate(tokens_list):
                B, N, C = tokens.shape
                H = W = int(math.sqrt(N))
                tokens_list[i] = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                           
        else:
            tokens_list = tokens_list
            pass                  
        
            
                                                         
        self._clamp_abs(self.alpha2.data, 1e-1)
        self._clamp_abs(self.alpha1.data, 1e-1)
        self._clamp_abs(self.alpha0.data, 1e-1) 

                                                                                                        
        x_emb,f0,f1,f2 = tokens_list[0],tokens_list[1],tokens_list[2],tokens_list[3]
        
                                                                                                           
                                                 

        t0 = self.deconv_o0(torch.cat([self.upsample1(f1),x_emb],dim=1))
        f0 = self.deconv_o0_f0(torch.cat([f0, self.mamba_blocks[0](t0)],dim=1))

        t1 = self.deconv_o1(torch.cat([self.upsample2(f2),self.downsample0(f0)],dim=1))
        f1 = self.deconv_o1_f1(torch.cat([f1, self.mamba_blocks[1](t1)],dim=1))

        t2 = self.downsample1(f1)
        f2 = self.deconv_o2_f2(torch.cat([f2, self.mamba_blocks[2](t2)],dim=1))

        tokens_spatial_list = [x_emb,f0,f1,f2]


        return tokens_spatial_list                                 

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()           
            data.abs_().clamp_(value)                                          
            data *= sign   


class ConvNextBlock(nn.Module):
\
\
\
\
\
\
\
\
\
       
    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path= 0.0):
        super().__init__()

                       
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_channel, padding_mode='reflect')                 
                                
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
                        
        self.pwconv1 = nn.Linear(in_channel, hidden_dim, bias=False)                                                          
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel, bias=False)    
                                                      
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
                
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)                                 
                                                                                         
        x = x.permute(0, 2, 3, 1)                                                                     
        x = self.norm(x)                      
        x = self.pwconv1(x)                             
        x = self.act(x)                          
        x = self.pwconv2(x)                              
        if self.gamma is not None:   
            x = self.gamma * x               
        x = x.permute(0, 3, 1, 2)                                                                      
        x = input + self.drop_path(x)                    

        return x                                   

class UnifiedTokenDecoder(nn.Module):
                                                                  
    def __init__(
        self,
        embed_dims=[96,192,384,768],
        base_scale_init=0.1,
        out_channels=6,
        context_chans=6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.context_chans = context_chans
        
        self.upsample1 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
                              
            nn.Conv2d(embed_dims[1], embed_dims[1] // 2, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(embed_dims[1] // 2, affine=True),               
            nn.GELU(),

            nn.Conv2d(embed_dims[1] // 2, embed_dims[1] // 2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[1] // 2, affine=True),
            nn.GELU(),          
        )
        self.convblock01 = nn.Sequential(
            ConvNextBlock(embed_dims[1]//2, 2*embed_dims[1]//2, embed_dims[1]//2, kernel_size=3, layer_scale_init_value=1.0, drop_path=0.05),
            ChannelAttention(dim=embed_dims[1]//2,num_heads=2,bias=True),
            )        


        self.upsample2 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
                              
            nn.Conv2d(embed_dims[2], embed_dims[2] // 2, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(embed_dims[2] // 2, affine=True),               
            nn.GELU(),

            nn.Conv2d(embed_dims[2] // 2, embed_dims[2] // 2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.InstanceNorm2d(embed_dims[2] // 2, affine=True),
            nn.GELU(),   
        )
        self.convblock12 = nn.Sequential(
            ConvNextBlock(embed_dims[2]//2, 2*embed_dims[2]//2, embed_dims[2]//2, kernel_size=3, layer_scale_init_value=1.0, drop_path=0.05),
            ChannelAttention(dim=embed_dims[2]//2,num_heads=2,bias=True),
            )
        

                                         
                                                                                
                                
                                                                                                
                                                                                
                        

                                                                                                                                        
                                                                 
                                     
           
                                           
                                                                                                                                               
                                                                           
               
        
        alpha_init_value = 0.7
        self.alpha0 = nn.Parameter(alpha_init_value * torch.ones((1, embed_dims[0], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha1 = nn.Parameter(alpha_init_value * torch.ones((1, embed_dims[1], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        self.alpha2 = nn.Parameter(alpha_init_value * torch.ones((1, embed_dims[2], 1, 1)),
                                   requires_grad=True) if alpha_init_value > 0 else None
        
                                                                                                                                                
        self.conv_o1 = nn.Conv2d(in_channels=embed_dims[2],out_channels=embed_dims[1],kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.conv_o0 = nn.Conv2d(in_channels=embed_dims[1],out_channels=embed_dims[0],kernel_size=3,stride=1,padding=1,padding_mode='reflect')

        
        
                   
        self.decoder = nn.Sequential(
                              
            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(96, affine=True),
            nn.GELU(),

                                                              

                                
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.GELU(),

            FrequencyProcessor(channels=64,int_size=128),

                  
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.GELU(),

                                      
            nn.Conv2d(32, out_channels, kernel_size=1, bias=True)
        )

                  
                                                                       
                                                                         
                                                                                 
        
        self.conv_out = nn.Conv2d(
            in_channels=out_channels + context_chans,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='reflect',
        )

    def forward(self, tokens_list, resident_tokens_list, x_in):
                                                    
                                     
        
        if tokens_list[0].ndim == 3:        
            for i, tokens in enumerate(tokens_list):
                B, N, C = tokens.shape
                H = W = int(math.sqrt(N))
                tokens_list[i] = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                           
        else:
            tokens_list = tokens_list
            pass                  

        if resident_tokens_list[0].ndim == 3:        
                                                               
            for i, res_tokens in enumerate(resident_tokens_list):
                B, N, C = res_tokens.shape
                H = W = int(math.sqrt(N))
                resident_tokens_list[i] = res_tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                           
        else:
            resident_tokens_list = resident_tokens_list 
            pass                  

        f0,f1,f2= tokens_list[1],tokens_list[2],tokens_list[3]
        r0,r1,r2 = resident_tokens_list[1],resident_tokens_list[2],resident_tokens_list[3]

        self._clamp_abs(self.alpha2,1e-1)
        self._clamp_abs(self.alpha1,1e-1)
        self._clamp_abs(self.alpha0,1e-1)

        o2 = self.alpha2*f2 + (1-self.alpha2)*r2                   

        o1 = self.alpha1*self.convblock12((self.conv_o1(torch.cat([f1 , self.upsample2(o2)],dim=1)))) + (1-self.alpha1)*r1                   

        o0 = self.alpha0*self.convblock01((self.conv_o0(torch.cat([f0 , self.upsample1(o1)],dim=1)))) + (1-self.alpha0)*r0                  

            
        delta = self.decoder(o0)                               
        
                                                               
                                                   
                                      
                                                     
                                                     
                                       

        if self.context_chans == x_in.shape[1]:
            context = x_in
        elif self.context_chans == 2 * x_in.shape[1]:
            context = torch.cat([x_in, x_in], dim=1)
        else:
            raise ValueError(
                f"context_chans={self.context_chans} is incompatible with x_in channels={x_in.shape[1]}"
            )

        output = self.conv_out(torch.cat([delta, context], dim=1))

        
        return output                               
    

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()           
            data.abs_().clamp_(value)                                          
            data *= sign    
