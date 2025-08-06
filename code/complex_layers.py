import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, inputs):
        real, imag = inputs
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        return real_out, imag_out


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, inputs):
        real, imag = inputs
        return self.bn_real(real), self.bn_imag(imag)


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, inputs):
        real, imag = inputs
        return F.relu(real), F.relu(imag)


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexMaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, inputs):
        real, imag = inputs
        return self.pool(real), self.pool(imag)


class ComplexFlatten(nn.Module):
    def __init__(self):
        super(ComplexFlatten, self).__init__()

    def forward(self, inputs):
        real, imag = inputs
        real_flat = real.view(real.size(0), -1)
        imag_flat = imag.view(imag.size(0), -1)
        return real_flat, imag_flat


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.real_linear = nn.Linear(in_features, out_features)
        self.imag_linear = nn.Linear(in_features, out_features)

    def forward(self, real_input, imag_input):
        real_out = self.real_linear(real_input) - self.imag_linear(imag_input)
        imag_out = self.real_linear(imag_input) + self.imag_linear(real_input)
        return real_out, imag_out


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.dropout_real = nn.Dropout(p)
        self.dropout_imag = nn.Dropout(p)

    def forward(self, inputs):
        real, imag = inputs
        return self.dropout_real(real), self.dropout_imag(imag)

