import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ms_ssim
except Exception:                                                   
    ms_ssim = None


class MaskAwareInpaintLoss(nn.Module):
\
\
\
\
\
\
       

    def __init__(
        self,
        l1_weight=1.0,
        msssim_weight=0.2,
        gradient_weight=0.2,
        laplacian_weight=0.1,
        boundary_weight=0.5,
        identity_weight=1.0,
        boundary_kernel=9,
        identity_kernel=9,
        eps=1e-6,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight
        self.boundary_weight = boundary_weight
        self.identity_weight = identity_weight
        self.boundary_kernel = boundary_kernel
        self.identity_kernel = identity_kernel
        self.eps = eps

        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel)

    def forward(self, fake_Ts, label1, input_image, rcmaps, fake_Rs=None, label2=None):
        pred = fake_Ts
        target = label1
        raw = input_image
        mask = self._prepare_mask(rcmaps, pred)

        reflection_l1 = self._masked_l1(pred, target, mask)
        msssim_loss = self._masked_msssim(pred, target, mask)
        gradient_loss = self._gradient_l1(pred, target, mask)
        laplacian_loss = self._laplacian_l1(pred, target, mask)

        boundary = self._boundary_band(mask, self.boundary_kernel)
        boundary_loss = self._masked_l1(pred, target, boundary)

        identity_keep = 1.0 - self._dilate(mask, self.identity_kernel)
        identity_loss = self._masked_l1(pred, raw, identity_keep)

        region_loss = self.l1_weight * reflection_l1
        detail_loss = self.gradient_weight * gradient_loss + self.laplacian_weight * laplacian_loss
        ssim_loss = self.msssim_weight * msssim_loss
        boundary_term = self.boundary_weight * boundary_loss
        identity_term = self.identity_weight * identity_loss

        total_loss = region_loss + detail_loss + ssim_loss + boundary_term + identity_term

        loss_table = {
            "reflection_l1_loss": reflection_l1,
            "msssim_loss": msssim_loss,
            "gradient_loss": gradient_loss,
            "laplacian_loss": laplacian_loss,
            "boundary_loss": boundary_loss,
            "identity_loss": identity_loss,
            "region_loss": region_loss,
            "detail_loss": detail_loss,
            "ssim_loss": ssim_loss,
            "boundary_term": boundary_term,
            "identity_term": identity_term,
            "total_loss": total_loss,
        }

        return loss_table, region_loss, detail_loss, ssim_loss, boundary_term, total_loss

    def _prepare_mask(self, mask, ref):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != 1:
            mask = mask.amax(dim=1, keepdim=True)
        mask = mask.to(device=ref.device, dtype=ref.dtype)
        if mask.shape[-2:] != ref.shape[-2:]:
            mask = F.interpolate(mask, size=ref.shape[-2:], mode="nearest")
        if float(mask.detach().amax().item()) > 1.0:
            mask = mask / 255.0
        return mask.clamp(0.0, 1.0)

    def _masked_l1(self, pred, target, mask):
        weight = mask.expand_as(pred)
        denom = weight.sum().clamp_min(self.eps)
        return ((pred - target).abs() * weight).sum() / denom

    def _masked_msssim(self, pred, target, mask):
        if ms_ssim is None or min(pred.shape[-2:]) < 160:
            return self._masked_l1(pred, target, mask)

        focus_pred = pred * mask + target * (1.0 - mask)
        focus_pred = focus_pred.clamp(0.0, 1.0)
        focus_target = target.clamp(0.0, 1.0)
        return 1.0 - ms_ssim(focus_pred, focus_target, data_range=1.0, size_average=True)

    def _gradient_l1(self, pred, target, mask):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        mask_dx = mask[:, :, :, 1:].expand_as(pred_dx)

        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        mask_dy = mask[:, :, 1:, :].expand_as(pred_dy)

        dx_loss = ((pred_dx - target_dx).abs() * mask_dx).sum() / mask_dx.sum().clamp_min(self.eps)
        dy_loss = ((pred_dy - target_dy).abs() * mask_dy).sum() / mask_dy.sum().clamp_min(self.eps)
        return 0.5 * (dx_loss + dy_loss)

    def _laplacian_l1(self, pred, target, mask):
        channels = pred.shape[1]
        kernel = self.laplacian_kernel_for(channels, pred)
        pred_lap = F.conv2d(pred, kernel, padding=1, groups=channels)
        target_lap = F.conv2d(target, kernel, padding=1, groups=channels)
        return self._masked_l1(pred_lap, target_lap, mask)

    def laplacian_kernel_for(self, channels, ref):
        return self.laplacian_kernel.to(device=ref.device, dtype=ref.dtype).repeat(channels, 1, 1, 1)

    def _dilate(self, mask, kernel_size):
        pad = kernel_size // 2
        return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)

    def _erode(self, mask, kernel_size):
        pad = kernel_size // 2
        return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)

    def _boundary_band(self, mask, kernel_size):
        return (self._dilate(mask, kernel_size) - self._erode(mask, kernel_size)).clamp(0.0, 1.0)


class CoreHaloReconstructionLoss(nn.Module):
                                                                                    

    def __init__(
        self,
        core_weight=1.0,
        halo_weight=0.5,
        identity_weight=0.1,
        halo_kernel=15,
        identity_kernel=9,
        eps=1e-6,
    ):
        super().__init__()
        self.core_weight = core_weight
        self.halo_weight = halo_weight
        self.identity_weight = identity_weight
        self.halo_kernel = halo_kernel
        self.identity_kernel = identity_kernel
        self.eps = eps

    def forward(self, fake_Ts, target, input_image, core_mask, fake_Rs=None, label2=None, halo_mask=None):
        pred = fake_Ts
        core = self._prepare_mask(core_mask, pred)
        halo = self._prepare_mask(halo_mask, pred) if halo_mask is not None else self._halo(core, self.halo_kernel)
        halo = (halo * (1.0 - core)).clamp(0.0, 1.0)
        identity_keep = 1.0 - self._dilate((core + halo).clamp(0.0, 1.0), self.identity_kernel)

        core_l1 = self._masked_l1(pred, target, core)
        halo_l1 = self._masked_l1(pred, target, halo)
        identity_l1 = self._masked_l1(pred, input_image, identity_keep)

        core_term = self.core_weight * core_l1
        halo_term = self.halo_weight * halo_l1
        identity_term = self.identity_weight * identity_l1
        total_loss = core_term + halo_term + identity_term

        loss_table = {
            "core_l1_loss": core_l1,
            "halo_l1_loss": halo_l1,
            "identity_l1_loss": identity_l1,
            "core_term": core_term,
            "halo_term": halo_term,
            "identity_term": identity_term,
            "total_loss": total_loss,
        }
        return loss_table, core_term, halo_term, identity_term, halo_l1, total_loss

    def _prepare_mask(self, mask, ref):
        if mask is None:
            return torch.zeros(ref.shape[0], 1, ref.shape[-2], ref.shape[-1], device=ref.device, dtype=ref.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != 1:
            mask = mask.amax(dim=1, keepdim=True)
        mask = mask.to(device=ref.device, dtype=ref.dtype)
        if mask.shape[-2:] != ref.shape[-2:]:
            mask = F.interpolate(mask, size=ref.shape[-2:], mode="nearest")
        if float(mask.detach().amax().item()) > 1.0:
            mask = mask / 255.0
        return mask.clamp(0.0, 1.0)

    def _masked_l1(self, pred, target, mask):
        weight = mask.expand_as(pred)
        denom = weight.sum().clamp_min(self.eps)
        return ((pred - target).abs() * weight).sum() / denom

    def _dilate(self, mask, kernel_size):
        pad = kernel_size // 2
        return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)

    def _halo(self, mask, kernel_size):
        return (self._dilate(mask, kernel_size) - mask).clamp(0.0, 1.0)
