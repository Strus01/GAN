from omegaconf import DictConfig
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, config: DictConfig):
        super(Discriminator, self).__init__()
        self.config = config

        self.conv_blocks = nn.ModuleList(
            [self.create_network_block(*self.config.conv_blocks[i]) for i in range(len(self.config.conv_blocks))]
        )
        self.output_layer = nn.Conv2d(**self.config.output_layer)
        self.sigmoid = nn.Sigmoid()

    def create_network_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(self.config.leaky_relu_negative_slope)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        pred = self.sigmoid(self.output_layer(x))
        return pred.view(len(pred), -1)
