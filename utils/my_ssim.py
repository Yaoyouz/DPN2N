import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from math import exp
import einops
from skimage import metrics
import numpy as np

def average(window_size, channel=1):
    return torch.ones((window_size, window_size, window_size), dtype=torch.float32,
                      requires_grad=False).contiguous() / (window_size ** 3)


def create_3d_window(size, sigma):
    kernel = np.fromfunction(
        lambda x, y, z: (1 / (2 * np.pi * sigma ** 2)) *
                        np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 + (z - (size - 1) / 2) ** 2) / (
                                2 * sigma ** 2)),
        (size, size, size)
    )
    kernel = kernel / np.sum(kernel)
    kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)
    return kernel


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# 构造损失函数用于网络训练,因为SSIM实际上效果不好，用MSSIM
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        # self.window = create_window(window_size, self.channel)
        self.window = create_window(11,1)

    def forward(self, img1, img2):
        channel = img1.size()[1]
        window = self.window.to(img1.device)
        ssim = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim




class SSIM_LOSS(nn.Module):
    def __init__(self):
        super(SSIM_LOSS, self).__init__()
        self.ssim = SSIM()

    def forward(self, img1, img2):
        loss = (1 - self.ssim(img1, img2)) / 2
        mean_loss = torch.mean(loss)
        return mean_loss


def calculate_simm_score(volume1, volume2):
    if torch.is_tensor(volume1):
        volume1 = (volume1.detach().cpu()).numpy()
    if torch.is_tensor(volume2):
        volume2 = (volume2.detach().cpu()).numpy()
    simm_score = 0
    for data1, data2 in zip(volume1, volume2):
        data1 = data1.squeeze()
        data2 = data2.squeeze()
        simm_score += metrics.structural_similarity(data1, data2, data_range=1)
    return simm_score


if __name__ == '__main__':
    x1 = torch.randn([2, 1, 128, 128])
    y1 = torch.randn([2, 1, 128, 128])
    max_x1 = torch.max(x1)
    min_x1 = torch.min(x1)
    x1 = (x1 - min_x1) / (max_x1 - min_x1)
    max_y1 = torch.max(y1)
    min_y1 = torch.min(y1)
    y1 = (y1 - min_y1) / (max_y1 - min_y1)
    # y1 = 1-x1
    ssim_loss = SSIM_LOSS()
    score = ssim_loss(x1, y1)
    print(score)
