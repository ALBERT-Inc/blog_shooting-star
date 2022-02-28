import torch
import torch.nn as nn
import numpy as np
from util import euler2matrix

# カメラ座標からレーザー座標への変換
class Cam2laser(nn.Module):
    def __init__(self, activation=lambda x: x):
        super().__init__()
        # Rotation
        self.r = \
            nn.Parameter(torch.Tensor(
                np.zeros(3)))
        # Translation
        self.t = \
            nn.Parameter(torch.Tensor(
                np.zeros(3)))
        self.activation = activation

    def forward(self, x):
        return self.activation(
            euler2matrix(x, self.r) + self.t)


class DestStep(nn.Module):
    def __init__(self, output_height, motor_step):
        super(DestStep, self).__init__()
        self.cam2laser_layer = Cam2laser()
        self.output_height = output_height
        self.motor_step = motor_step

    def forward(self, x):
        h = self.cam2laser_layer(x)[0]
        x = h[:, 0]
        y = h[:, 1]
        z = h[:, 2]
        y_ = torch.sqrt((y + self.output_height)**2 + z**2) + self.output_height
        deg_1 = torch.arcsin(x / torch.sqrt(y_ ** 2 + x ** 2)) / 2
        x_ = torch.sqrt(x**2 + z**2)
        deg_2 = torch.arcsin((y + self.output_height) /
                             torch.sqrt(x_ ** 2 + (y + self.output_height) ** 2)) / 2
        deg_1 = (deg_1 / (2 * (torch.pi))) * self.motor_step
        deg_2 = (deg_2 / (2 * (torch.pi))) * self.motor_step
        h = torch.vstack((deg_1, deg_2))
        h = torch.t(h)
        return h