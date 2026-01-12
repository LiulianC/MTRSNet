import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding=None, dilation=1, norm=None, act=None, bias=False):
        super(Conv2DLayer, self).__init__()  
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2 
        self.add_module('conv2d', nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation=dilation, bias=bias, padding_mode='reflect')) 
        if norm is not None: 
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)

class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.se = nn.Sequential( 
            nn.Linear(channel, channel // reduction),
            nn.LayerNorm(channel//reduction),
            nn.ReLU(inplace=True), 
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y
    

class ResidualBlock(nn.Module):
    def __init__(self, channel, norm=nn.InstanceNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=1, act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.se_layer = None
        self.res_scale = res_scale 
        if se_reduction is not None: 
            self.se_layer = SElayer(channel, se_reduction)

    def forward(self, x):
        res = x 
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_layer:
            x = self.se_layer(x) 
        x = x * self.res_scale 
        out = x + res 
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_1 = nn.Conv2d(channel, channel // reduction, 1, bias=True)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Conv2d(channel // reduction, channel, 1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_output = self.fc_2(self.relu(self.fc_1(self.avg_pool(x))))
        max_output = self.fc_2(self.relu(self.fc_1(self.max_pool(x))))
        out = avg_output + max_output
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7.'
        padding_size = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, padding=padding_size, bias=False, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True) 

        pool_out = torch.cat([avg_out, max_out], dim=1) 
        x = self.conv(pool_out) 
        return self.sigmoid(x) 

class CBAMlayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMlayer, self).__init__()
        self.channel_layer = ChannelAttention(channel, reduction)
        self.spatial_layer = SpatialAttention()

    def forward(self, x):
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x

class ResidualCbamBlock(nn.Module):

    def __init__(self, channel, norm=nn.InstanceNorm2d, dilation=1, bias=False, cbam_reduction=None, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam_layer = None
        if cbam_reduction is not None:
            self.cbam_layer = CBAMlayer(channel, cbam_reduction)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam_layer:
            x = self.cbam_layer(x)

        out = x + res
        return out
    
class LaplacianPyramid(nn.Module):

    def __init__(self, device, dim=3):
        super(LaplacianPyramid, self).__init__()

        self.channel_dim = dim
        laplacian_kernel = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

        laplacian_kernel = np.repeat(laplacian_kernel[None, None, :, :], dim, 0) 


        self.kernel = torch.nn.Parameter(torch.FloatTensor(laplacian_kernel))
        self.register_buffer('kernel_init', torch.FloatTensor(laplacian_kernel).clone())

        epsilon = 0.05
        with torch.no_grad():
            self.kernel.data.clamp_(self.kernel_init - epsilon, self.kernel_init + epsilon)


    def forward(self, x):
        x0 = F.interpolate(x, scale_factor=0.125, mode='bilinear')
        x1 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        lap_0 = F.conv2d(x0, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_1 = F.conv2d(x1, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_2 = F.conv2d(x2, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_3 = F.conv2d(x, self.kernel, groups=self.channel_dim, padding=1, stride=1, dilation=1)
        lap_0 = F.interpolate(lap_0, scale_factor=8, mode='bilinear')
        lap_1 = F.interpolate(lap_1, scale_factor=4, mode='bilinear')
        lap_2 = F.interpolate(lap_2, scale_factor=2, mode='bilinear')

        return torch.cat([lap_0, lap_1, lap_2, lap_3], 1) 

class LRM(nn.Module):

    def __init__(self, device):
        super(LRM, self).__init__()

        self.lap_pyramid = LaplacianPyramid(device, dim=6) 

        self.det_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )

        self.det_conv1 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv2 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv3 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4_1 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)
        self.det_conv4_2 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1)

        self.det_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
            )

        self.det_conv6 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv7 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv8 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv9 = ResidualBlock( channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv10 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv11 = ResidualBlock(channel=32, norm=None, se_reduction=2, res_scale=0.1, act=nn.PReLU())

        self.p_relu = nn.PReLU()
        self.relu = nn.ReLU()

        self.det_conv_mask0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size= 3, stride= 1, padding= 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )        

    def forward(self, I):

        x = torch.cat([I, I], 1) 
        lap = self.lap_pyramid(x) 

        x = self.det_conv0(x) 
        x = F.relu(self.det_conv1(x)) 
        x = F.relu(self.det_conv2(x)) 
        x = F.relu(self.det_conv3(x)) 
        x = F.relu(self.det_conv4(x)) 
        x = F.relu(self.det_conv4_1(x)) 
        x = F.relu(self.det_conv4_2(x)) 

        lap = self.det_conv5(lap) 
        lap = self.p_relu(self.det_conv6(lap)) 
        lap = self.p_relu(self.det_conv7(lap)) 
        lap = self.p_relu(self.det_conv8(lap)) 
        c_map = self.det_conv_mask0(lap) 
        lap = self.p_relu(self.det_conv9(lap))  
        lap = self.p_relu(self.det_conv10(lap)) 
        lap = self.p_relu(self.det_conv11(lap)) 

        lap = lap * c_map 

        x = torch.cat([x, lap], 1)  
        return x,c_map








import torch.fft as fft
import math
from einops import rearrange


def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1 * torch.exp(1j * (fft_.angle())))
    return fft_.real

class Toning(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(Toning, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):
        x1 = inv_mag(x)
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Mapping(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=3, res=True):
        """
        Parameters:
            in_features (int): Number of input features (channels).
            hidden_features (int): Number of features in hidden layers.
            hidden_layers (int): Number of hidden layers.
            out_features (int): Number of output features (channels).
            res (bool): Whether to use residual connections.
        """
        super(Mapping, self).__init__()

        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())

        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, inp):
        original_shape = inp.shape
        inp = inp.view(-1, inp.shape[1])

        output = self.net(inp)

        if self.res:
            output = output + inp
            output = torch.clamp(output, 0., 1.)

        output = output.view(original_shape)

        return output

class FrequencyProcessor(nn.Module):
    def __init__(self, channels=3, int_size=64):
        super(FrequencyProcessor, self).__init__()
        self.identity1 = nn.Conv2d(channels, channels, 1)
        self.identity2 = nn.Conv2d(channels, channels, 1)

        self.conv_f1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.map = Mapping(in_features=channels, out_features=channels, hidden_features=int_size, hidden_layers=5)
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.tone = Toning(channels)

    def forward(self, x):
        out = self.identity1(x)

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft = F.gelu(self.conv_f1(x_fft))
        x_fft = self.map(x_fft)
        x_reconstructed = fft.ifftn(x_fft, dim=(-2, -1)).real
        x_reconstructed += self.identity2(x)

        f_out = self.fuse(torch.cat([out, x_reconstructed], dim=1))

        return self.tone(f_out)



class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(c / self.num_heads))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    






