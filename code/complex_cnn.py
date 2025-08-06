from torch import nn
from torch.nn.functional import relu
import torch.nn.functional as F
import torch


class ComplexCNN(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(ComplexCNN, self).__init__()
        self.in_shape = in_shape
        in_channels = in_shape[2]
        self.num_classes = num_classes

        self.conv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.norm2d_1 = nn.BatchNorm2d(8)
        self.maxPool2d_1 = nn.MaxPool2d(2)

        self.conv2d_2 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=5, padding=2)
        self.norm2d_2 = nn.BatchNorm2d(2)
        self.maxPool2d_2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)

        self.flatten_1 = nn.Flatten(2)
        self.flatten_2 = nn.Flatten()

        self.complex_fc_1 = m_Linear(2 * (self.in_shape[-1] // 4) * (self.in_shape[-2] // 4), 128)
        self.complex_fc_2 = m_Linear(128, 64)


        self.fc_1 = nn.Linear(196, 20)
        self.fc_2 = nn.Linear(20, num_classes)

    def forward(self, x):
        """
        :param x: (batch_size, 2, channel, window_size, sub-carrier), 2表示实部和虚部
        :return:
        """
        t, _, c, h, w = x.shape # (t, 2, c, h, w)

        # 复数卷积（本质还是实值卷积）
        x = x.reshape(-1, c, h, w)            # (t, 2, c, h, w) -> (t*2, c, h, w)
        output = self.conv2d_1(x)             # (t*2, c, h, w) -> (t*2, 8, h, w)
        output = relu(self.norm2d_1(output))        # (t*2, 8, h, w) -> (t*2, 8, h, w)
        output = self.maxPool2d_1(output)     # (t*2, 8, h, w) -> (t*2, 8, h//2, w//2)

        output = self.conv2d_2(output)        # (t*2, 8, h//2, w//2) -> (t*2, 2, h//2, w//2)
        output = relu(self.norm2d_2(output))  # (t*2, 2, h//2, w//2) -> (t*2, 2, h//2, w//2)
        output = self.maxPool2d_2(output)     # (t*2, 2, h//2, w//2) -> (t*2, 2, h//4, w//4)
        output = self.dropout(output)         # (t*2, 8, h//4, w//4) -> (t*2, 2, h//4, w//4)
        output = output.reshape((-1, 2) + output.shape[1:]) # (t*2, 2, h//4, w//4) -> (t, 2, 2, h//4, w//4)

        # 复数全连接
        output = self.flatten_1(output)       # (t, 2, 2, h//4, w//4) -> (t, 2, h * w // 8)
        output = relu(self.complex_fc_1(output))    # (t, 2, h * w // 8) -> (t, 2, 128)
        output = self.dropout(output)         # (t, 2, 128) -> (t, 2, 128)
        output = relu(self.complex_fc_2(output))    # (t, 2, 128) -> (t, 2, 64)
        output = self.dropout(output)         # (t, 2, 64) -> (t, 2, 64)

        # output = self.flatten_2(output)       # (t, 2, 64) -> (t, 128)
        # output = relu(self.fc_1(output))      # (t, 128) -> (t, 20)
        # output = self.fc_2(output)            # (t, 20) -> (t, 2)

        return F.softmax(output, dim=1)


class m_Linear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        # Creation
        self.weights_real = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.weights_imag = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Initialization
        nn.init.xavier_uniform_(self.weights_real, gain=1)
        nn.init.xavier_uniform_(self.weights_imag, gain=1)
        nn.init.zeros_(self.bias)

    def swap_real_imag(self, x):
        # [@, 2, a]
        # [real, imag] => [-1*imag, real]
        _, _, a = x.shape
        h = x  # [@, 2, a]
        h = h.flip(dims=[-2])  # [@, 2, a]  [real, imag]=>[imag, real]
        imag_real = torch.stack((-1 * torch.ones(a), torch.ones(a)), dim=0).cuda()
        h = h * imag_real # [@, 2, a] [imag, real]=>[-1*imag, real]
        return h

    def forward(self, x):
        # x: (t, 2, a)
        h = x
        h1 = torch.matmul(h, self.weights_real)
        h2 = torch.matmul(h, self.weights_imag)
        h2 = self.swap_real_imag(h2)
        h = h1 + h2
        h = torch.add(h, self.bias)
        return h


if __name__ == '__main__':
    device = torch.device("cuda:0")
    x = torch.randn(16, 2, 1, 30, 28).to(device)
    model = ComplexCNN((16, 2, 1) + (30, 28),  2).to(device)
    model.forward(x)




