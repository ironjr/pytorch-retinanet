'''RetinaFPN in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    '''3x3 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class Bottleneck(nn.Module):
    '''Bottleneck block for ResNet-50 and ResNet-101 backbone'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPNResNet(nn.Module):
    '''Feature Pyramid Network based on ResNet backbone'''

    def __init__(self, block, num_blocks, num_features=256):
        super(FPNResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Additional layers introduced in FPN
        # Those layers should have bias term since no batchnorm follows after
        # The layer configuration follows Focal Loss paper
        self.conv6 = conv3x3(2048, num_features, stride=2, bias=True)
        self.conv7 = conv3x3(num_features, num_features, stride=2, bias=True)

        # Lateral layers
        self.latlayer1 = conv1x1(2048, num_features, bias=True)
        self.latlayer2 = conv1x1(1024, num_features, bias=True)
        self.latlayer3 = conv1x1( 512, num_features, bias=True)

        # Top-down layers
        self.toplayer1 = conv3x3(num_features, num_features, bias=True)
        self.toplayer2 = conv3x3(num_features, num_features, bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return y + F.interpolate(x, size=(H, W), mode='bilinear',
                align_corners=False)

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7


def FPNResNet50(num_features=256):
    return FPNResNet(Bottleneck, [3, 4, 6, 3], num_features=num_features)

def FPNResNet101(num_features=256):
    return FPNResNet(Bottleneck, [3, 4, 23, 3], num_features=num_features)

def FPNResNet152(num_features=256):
    return FPNResNet(Bottleneck, [3, 8, 36, 3], num_features=num_features)


def test():
    net = FPNResNet50()
    fms = net(Variable(torch.randn(1, 3, 600, 300)))
    for fm in fms:
        print(fm.size())

# test()
