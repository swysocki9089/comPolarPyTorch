import torch
from torch.nn import Module
import torch.nn.functional as nnf
from polarComplexFunctions import *
from complexPyTorch import complexLayers as cl


class PolarDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, polar):
        if self.training:
            return polar_dropout(polar, p=self.p)
        else:
            return polar


class PolarDropout2d(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, polar):
        if self.training:
            return polar_dropout2d(polar, p=self.p)
        else:
            return polar


class PolarAvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, polar):
        return polar_avg_pool2d(
            polar,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override
        )


class PolarMaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, polar):
        return polar_max_pool2d(
            polar,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices
        )


class PolarReLU(Module):
    def forward(self, polar):
        return polar_relu(polar)


class PolarSigmoid(Module):
    def forward(self, polar):
        return polar_sigmoid(polar)


class PolarTanh(Module):
    def forward(self, polar):
        return polar_tanh(polar)

class PolarConv2d(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_mag = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding, dilation, groups, bias)
        self.conv_phase = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride, padding, dilation, groups, bias)

    def forward(self, polar):
        new_mag = self.conv_mag(polar.get_magnitude())
        new_phase = self.conv_phase(polar.get_phase())
        return ComPolar64(new_mag, new_phase)
    
class PolarLinear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_mag = torch.nn.Linear(in_features, out_features)
        self.fc_phase = torch.nn.Linear(in_features, out_features)

    def forward(self, polar):
        new_mag = self.fc_mag(polar.get_magnitude())
        new_phase = self.fc_phase(polar.get_phase())
        return ComPolar64(new_mag, new_phase)
    
class NaivePolarBatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn_mag = torch.nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_phase = torch.nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, polar):
        normed_mag = self.bn_mag(polar.get_magnitude())
        normed_phase = self.bn_phase(polar.get_phase())
        return ComPolar64(normed_mag, normed_phase)

class NaivePolarBatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn_mag = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_phase = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, polar):
        normed_mag = self.bn_mag(polar.get_magnitude())
        normed_phase = self.bn_phase(polar.get_phase())
        return ComPolar64(normed_mag, normed_phase)

class PolarConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0, output_padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_tran_mag = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode
        )
        self.conv_tran_phase = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode
        )

    def forward(self, polar):
        new_mag = self.conv_tran_mag(polar.get_magnitude())
        new_phase = self.conv_tran_phase(polar.get_phase())
        return ComPolar64(new_mag, new_phase)

class PolarBatchNorm2d(Module):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.cartesian_bn = cl.ComplexBatchNorm2d(num_features, **kwargs)

    def forward(self, polar):
        # convert polar to cartesian tensor
        x_cartesian = polar.to_cartesian()
        # apply cartesian batch norm
        bn_out = self.cartesian_bn(x_cartesian)
        # convert back to polar
        return ComPolar64.from_cartesian(bn_out)
