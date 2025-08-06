import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexMaxPool2d, ComplexFlatten, ComplexLinear, ComplexDropout
from modules.transformer import TransformerEncoder

class ComplexTransformerClassifier(nn.Module):
    def __init__(self, input_shape, embed_dim, hidden_size, output_dim, num_heads,
                 attn_dropout, relu_dropout, res_dropout, out_dropout, layers, attn_mask=False):
        """
        input_shape: (H, W) 输入张量的高和宽
        """
        super(ComplexTransformerClassifier, self).__init__()

        self.conv = nn.Sequential(
            ComplexConv2d(1, 16, kernel_size=3, stride=1, padding=1),
            ComplexBatchNorm2d(16),
            ComplexReLU(),
            ComplexMaxPool2d(2, 2),
            ComplexConv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            ComplexMaxPool2d(2, 2),
            ComplexConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(2, 2),
            ComplexConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ComplexBatchNorm2d(128),
            ComplexReLU(),
            ComplexMaxPool2d(2, 2),
            ComplexDropout(p=0.5),  # 添加 ComplexDropout
            ComplexFlatten()
        )

        # 计算展平后的维度（H 和 W 会缩小 2^4 = 16 倍）
        h, w = input_shape
        self.flat_size = (h // 16) * (w // 16) * 128

        # 嵌入投影层
        self.proj = ComplexLinear(self.flat_size, embed_dim)

        # Transformer 编码器
        self.trans = TransformerEncoder(embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        layers=layers,
                                        attn_dropout=attn_dropout,
                                        relu_dropout=relu_dropout,
                                        res_dropout=res_dropout,
                                        attn_mask=attn_mask)

        # 分类头
        self.out_fc1 = nn.Linear(embed_dim * 2, hidden_size)
        self.out_fc2 = nn.Linear(hidden_size, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, x):
        """
        x shape: [B, 2, H, W]
        """
        input_a = x[:, 0:1, :, :]  # [B, 1, H, W] 实部
        input_b = x[:, 1:2, :, :]  # [B, 1, H, W] 虚部

        input_a, input_b = self.conv((input_a, input_b))  # 复值CNN模块
        input_a = input_a.unsqueeze(0)  # 转换为 [1, B, D]
        input_b = input_b.unsqueeze(0)

        input_a, input_b = self.proj(input_a, input_b)
        h_as, h_bs = self.trans(input_a, input_b)
        h_concat = torch.cat([h_as, h_bs], dim=-1)

        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(h_concat))))
        output = output.squeeze(0)  # 去除时间维度
        return output

