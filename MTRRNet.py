import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class MTRRNet(nn.Module):
    def __init__(self, mask_threshold=0.5, fallback_bright_threshold=0.88, fallback_sat_threshold=0.22):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.fallback_bright_threshold = fallback_bright_threshold
        self.fallback_sat_threshold = fallback_sat_threshold
        self._init_token_model()

    def _init_token_model(self):
        from MTRR_token_modules import Encoder, SubNet, UnifiedTokenDecoder, init_all_weights

        self.use_rev = True
        self.token_encoder = Encoder(
            in_chans=4,
            embed_dim=96,
            mamba_blocks=[10, 10, 10, 10],
            swin_blocks=[4, 4, 4, 4],
            drop_branch_prob=0.2,
        )
        self.token_decoder0 = UnifiedTokenDecoder(
            embed_dims=[96, 192, 384, 768],
            base_scale_init=0.1,
            out_channels=3,
            context_chans=4,
        )
        self.token_subnet1 = SubNet(
            embed_dims=[96, 192, 384, 768],
            mam_blocks=[6, 6, 6, 6],
            use_rev=self.use_rev,
        )
        self.token_decoder1 = UnifiedTokenDecoder(
            embed_dims=[96, 192, 384, 768],
            base_scale_init=0.1,
            out_channels=3,
            context_chans=4,
        )
        self.token_subnet2 = SubNet(
            embed_dims=[96, 192, 384, 768],
            mam_blocks=[6, 6, 6, 6],
            use_rev=self.use_rev,
        )
        self.token_decoder2 = UnifiedTokenDecoder(
            embed_dims=[96, 192, 384, 768],
            base_scale_init=0.1,
            out_channels=3,
            context_chans=4,
        )
        self.token_subnet3 = SubNet(
            embed_dims=[96, 192, 384, 768],
            mam_blocks=[6, 6, 6, 6],
            use_rev=self.use_rev,
        )
        self.token_decoder3 = UnifiedTokenDecoder(
            embed_dims=[96, 192, 384, 768],
            base_scale_init=0.1,
            out_channels=3,
            context_chans=4,
        )
        init_all_weights(self)

    def _infer_core_mask(self, x_in):
        rgb_max = x_in.amax(dim=1, keepdim=True)
        rgb_min = x_in.amin(dim=1, keepdim=True)
        saturation = rgb_max - rgb_min
        mask = (rgb_max > self.fallback_bright_threshold) & (saturation < self.fallback_sat_threshold)
        return mask.to(dtype=x_in.dtype)

    def _normalize_core_mask(self, x_in, m_core):
        if m_core is None:
            return self._infer_core_mask(x_in)
        if m_core.ndim == 3:
            m_core = m_core.unsqueeze(1)
        if m_core.ndim != 4:
            raise ValueError(f"M_core must be [B,1,H,W], [B,3,H,W], or [B,H,W], got {tuple(m_core.shape)}")
        m_core = m_core.to(device=x_in.device, dtype=x_in.dtype)
        if m_core.shape[1] != 1:
            m_core = m_core.amax(dim=1, keepdim=True)
        if m_core.shape[-2:] != x_in.shape[-2:]:
            m_core = F.interpolate(m_core, size=x_in.shape[-2:], mode="nearest")
        if float(m_core.detach().amax().item()) > 1.0:
            m_core = m_core / 255.0
        return (m_core.clamp(0.0, 1.0) > self.mask_threshold).to(dtype=x_in.dtype)

    def _compose_masked_output(self, raw_patch, i_clean, m_core):
        o_clean = torch.sigmoid(raw_patch)
        y = i_clean + o_clean * m_core
        return y, o_clean

    def _decode_from_tokens(self, tokens_list, resident_tokens_list, decoder_input):
        raw0 = self.token_decoder0(tokens_list, resident_tokens_list, decoder_input)
        tokens_list = self.token_subnet1(tokens_list)
        raw1 = self.token_decoder1(tokens_list, resident_tokens_list, decoder_input)
        tokens_list = self.token_subnet2(tokens_list)
        raw2 = self.token_decoder2(tokens_list, resident_tokens_list, decoder_input)
        tokens_list = self.token_subnet3(tokens_list)
        raw3 = self.token_decoder3(tokens_list, resident_tokens_list, decoder_input)
        return [raw0, raw1, raw2, raw3]

    def _token_mask_to_image_mask(self, token_mask, token_grid, image_size, dtype):
        batch, tokens = token_mask.shape
        height, width = token_grid
        if tokens != height * width:
            raise ValueError(f"token_mask has {tokens} tokens but token_grid={token_grid}")
        mask = token_mask.view(batch, 1, height, width).to(dtype=dtype)
        return F.interpolate(mask, size=image_size, mode="nearest")

    def forward(self, x_in, m_core=None):
        m_core = self._normalize_core_mask(x_in, m_core)
        i_clean = x_in * (1.0 - m_core)
        repair_input = torch.cat([i_clean, m_core], dim=1)
        tokens_list = self.token_encoder(repair_input)
        resident_tokens_list = tokens_list
        raw_outs = self._decode_from_tokens(tokens_list, resident_tokens_list, repair_input)
        out0, patch0 = self._compose_masked_output(raw_outs[0], i_clean, m_core)
        out1, patch1 = self._compose_masked_output(raw_outs[1], i_clean, m_core)
        out2, patch2 = self._compose_masked_output(raw_outs[2], i_clean, m_core)
        out3, patch3 = self._compose_masked_output(raw_outs[3], i_clean, m_core)
        return [out0, out1, out2, out3], m_core, [patch0, patch1, patch2, patch3]

    def forward_pretrain(self, x_clean, mask_ratio=0.2, token_mask=None):
        zero_mask = torch.zeros(
            x_clean.shape[0],
            1,
            x_clean.shape[-2],
            x_clean.shape[-1],
            device=x_clean.device,
            dtype=x_clean.dtype,
        )
        repair_input = torch.cat([x_clean, zero_mask], dim=1)
        tokens_list, token_mask, token_grid = self.token_encoder(
            repair_input,
            token_mask=token_mask,
            mask_ratio=mask_ratio,
            return_token_mask=True,
        )
        raw_outs = self._decode_from_tokens(tokens_list, tokens_list, repair_input)
        pred_patches = [torch.sigmoid(x) for x in raw_outs]
        image_mask = self._token_mask_to_image_mask(token_mask, token_grid, x_clean.shape[-2:], x_clean.dtype)
        outs = [x_clean * (1.0 - image_mask) + pred * image_mask for pred in pred_patches]
        return {
            "outs": outs,
            "raw_outs": raw_outs,
            "pred_patches": pred_patches,
            "token_mask": token_mask,
            "token_grid": token_grid,
            "image_mask": image_mask,
        }


class MTRREngine(nn.Module):
    def __init__(self, opts=None, device="cuda"):
        super().__init__()
        self.device = device
        self.opts = opts
        self.visual_names = ["fake_Ts", "fake_Rs", "o_cleans", "c_map", "I", "Ic", "T", "R"]
        self.fake_Ts = [None] * 4
        self.fake_Rs = [None] * 4
        self.o_cleans = [None] * 4
        self.netG_T = MTRRNet().to(device)

    def get_current_visuals(self):
        visual_result = OrderedDict()
        for name in self.visual_names:
            visual_result[name] = getattr(self, name)
        return visual_result

    def set_input(self, input):
        self.I = input["input"].to(self.device)
        self.T = input["target_t"].to(self.device)
        self.R = input["target_r"].to(self.device)
        self.M_core = self._get_input_mask(input)

    def _get_input_mask(self, input):
        for key in ("M_repair", "M_core", "m_core", "core_mask", "mask", "c_map"):
            if key in input and input[key] is not None:
                return input[key].to(self.device)
        if "target_r" in input and input["target_r"] is not None:
            target_r = input["target_r"].to(self.device)
            if target_r.numel() > 0 and float(target_r.detach().abs().amax().item()) > 0.0:
                threshold = getattr(self.opts, "derived_mask_threshold", 0.05) if self.opts is not None else 0.05
                return (target_r.abs().amax(dim=1, keepdim=True) > threshold).float()
        return None

    def forward(self, input=None):
        self.Ic = self.I
        self.outs, self.c_map, self.o_cleans = self.netG_T(self.Ic, self.M_core)
        for idx, pred in enumerate(self.outs):
            self.fake_Ts[idx] = pred
            self.fake_Rs[idx] = (self.Ic - pred) * self.c_map

    def inference(self):
        self.forward()

    def count_parameters(self):
        total = sum(p.numel() for p in self.netG_T.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total:,}")
