"""
Image encoder: grayscale-adapted, ImageNet-pretrained ResNet-50 with a
soft-attention region branch (manuscript Sec. 3.1.1). Produces both
region-level features (for word-region alignment) and a global sentence-
level image embedding.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SoftAttention(nn.Module):
    def __init__(self, in_groups, m_heads, in_channels):
        super(SoftAttention, self).__init__()
        self.learnable_scalar = nn.Parameter(torch.rand(1))
        self.conv3d = nn.Conv3d(
            in_channels  = in_groups,
            out_channels = m_heads,
            kernel_size  = (in_channels, 1, 1),
            stride       = (in_channels, 1, 1)
        )
        self.lrelu   = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        c = torch.unsqueeze(x, 1)
        c = self.conv3d(c)
        c = self.lrelu(c)
        c = c.squeeze(2)
        c = c.view(c.shape[0], c.shape[1], h * w)
        c = self.softmax(c)
        c = c.view(c.shape[0], c.shape[1], h, w)
        attn_maps  = torch.unsqueeze(c.sum(1), 1)
        importance = x * attn_maps
        out = x + importance * self.learnable_scalar.expand_as(importance)
        return out, attn_maps, self.learnable_scalar


class ImageEncoder(nn.Module):
    def __init__(self, output_channels=512, pretrained=True):
        super(ImageEncoder, self).__init__()

        weights = 'IMAGENET1K_V2' if pretrained else None
        resnet  = tvm.resnet50(weights=weights)

        # adapt first conv for grayscale
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride      = old_conv.stride,
            padding     = old_conv.padding,
            bias        = False
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(
                    old_conv.weight.sum(dim=1, keepdim=True)
                )
        resnet.conv1 = new_conv

        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3   # 1024 channels
        self.layer4 = resnet.layer4   # 2048 channels

        self.sa_l3        = SoftAttention(in_groups=1, m_heads=16, in_channels=1024)
        self.region        = conv1x1(1024, output_channels)
        self.avgpool        = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout        = nn.Dropout(p=0.3)
        self.global_feats   = nn.Linear(2048, output_channels)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x, attn_maps_l3, scalar_l3 = self.sa_l3(x)
        region_feat = self.region(x)

        x = self.layer4(x)
        x = self.avgpool(x)
        z = self.dropout(torch.flatten(x, 1))
        global_feat = self.global_feats(z)

        return (region_feat, global_feat, attn_maps_l3, scalar_l3, attn_maps_l3, scalar_l3)

