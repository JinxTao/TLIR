import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import torchvision.models as models
import math
import numpy as np
import torch

class VGG16(nn.Module):
    """
    VGG16 神经网络模型。
    """

    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.layer2 = nn.Sequential(*list(self.vgg.children())[0:4])
        self.layer9 = nn.Sequential(*list(self.vgg.children())[0:16])
        self.layer12 = nn.Sequential(*list(self.vgg.children())[0:23])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        前向传递。

        参数:
            x (torch.Tensor): 输入图像，大小为 [batch_size, channels, height, width]。

        返回:
            list: VGG16 网络的第2、9、12层的输出，大小分别为 [batch_size, 64, height/2, width/2]、[batch_size, 128, height/4, width/4] 和 [batch_size, 256, height/8, width/8]。
        """
        out2 = self.layer2(x)
        out9 = self.layer9(x)
        out12 = self.layer12(x)
        return [out2, out9, out12]

def ssim_loss(img1, img2, device, window_size=11, size_average=True, sigma=1.5):
    """
    计算两个图像的 SSIM 损失函数。

    参数:
        img1 (torch.Tensor): 第一个图像，大小为 [batch_size, channels, height, width]。
        img2 (torch.Tensor): 第二个图像，大小为 [batch_size, channels, height, width]。
        window_size (int): 计算 SSIM 的窗口大小，默认为 11。
        size_average (bool): 是否对 batch 中每个图像计算平均值，默认为 True。
        sigma (float): SSIM 窗口的高斯权重参数，默认为 1.5。

    返回:
        torch.Tensor: 两个图像之间的 SSIM 损失值。
    """

    # 根据窗口大小生成高斯权重。
    gaussian = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gaussian = gaussian / gaussian.sum()
    gaussian = gaussian.to(device)

    # 对高斯权重进行卷积。
    gaussian = gaussian.view(1, 1, window_size, 1).repeat(1, 1, 1, window_size)
    gaussian_filter = torch.nn.Conv2d(1, 1, kernel_size=window_size, stride=1, padding=0, bias=False).to(device)
    gaussian_filter.weight.data = gaussian
    gaussian_filter.weight.requires_grad = False

    # 计算均值和标准差。
    mu1 = F.conv2d(img1, weight=gaussian_filter.weight, stride=1, padding=0, groups=img1.shape[1]).to(device)
    mu2 = F.conv2d(img2, weight=gaussian_filter.weight, stride=1, padding=0, groups=img2.shape[1]).to(device)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, weight=gaussian_filter.weight, stride=1, padding=0, groups=img1.shape[1]).to(device) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, weight=gaussian_filter.weight, stride=1, padding=0, groups=img2.shape[1]).to(device) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, weight=gaussian_filter.weight, stride=1, padding=0, groups=img1.shape[1]).to(device) - mu1_mu2

    # 计算 SSIM。
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    if size_average:
        return torch.mean(1 - ssim_map)
    else:
        return torch.sum(1 - ssim_map)


class CombinedLoss(nn.Module):
    """
    结合 SSIM 损失和感知损失的总损失函数。
    """

    def __init__(self, layers=[2, 9, 12], device='cuda'):
        super(CombinedLoss, self).__init__()
        self.layers = layers
        self.device = device

    def forward(self, img1, img2):
        """
        计算两个图像之间的总损失。

        参数:
            img1 (torch.Tensor): 第一个图像，大小为 [batch_size, channels, height, width]。
            img2 (torch.Tensor): 第二个图像，大小为 [batch_size, channels, height, width]。

        返回:
            torch.Tensor: 两个图像之间的总损失值。
        """

        # 计算 SSIM 损失
        ssim_loss_value = ssim_loss(img1, img2,self.device)

        # 计算感知损失
        vgg16 = VGG16().to(self.device)
        # img1 = np.array(img1.cpu())
        # img2 = np.array(img2.cpu())
        # img1 = np.concatenate([img1,img1,img1],axis=1)
        # img2 = np.concatenate([img2,img2,img2],axis=1)
        # img2 = torch.tensor(img2).to(self.device)
        # img1 = torch.tensor(img1).to(self.device)
        img1 = torch.cat([img1,img1,img1],1)
        img2 = torch.cat([img2,img2,img2],1)
        img1_features = vgg16(img1.to(self.device))
        img2_features = vgg16(img2.to(self.device))
        loss = 0
        for i in range(3):
            loss += F.mse_loss(img1_features[i], img2_features[i])
        perception_loss_value = loss / len(self.layers)

        # 计算总损失
        loss = ssim_loss_value *  perception_loss_value
        return loss

