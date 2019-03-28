import torch
import torch.nn as nn

from fpn import conv1x1, conv3x3, FPNResNet50, FPNResNet101, FPNResNet152
from torch.autograd import Variable


class RetinaNet(nn.Module):

    # In Focal Loss paper, anchors have
    # three aspect ratios {1:2, 1:1, 2:1}
    # three sizes         {2^0, 2^(1/3), 2^(2/3)}
    num_anchors = 9
    
    def __init__(self, backbone='resnet101', num_classes=80, num_features=256):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes

        # Apply proper backbone network
        if backbone == 'resnet50':
            self.fpn = FPNResNet50(num_features=num_features)
        elif backbone == 'resnet101':
            self.fpn = FPNResNet101(num_features=num_features)
        elif backbone == 'resnet152':
            self.fpn = FPNResNet152(num_features=num_features)
        else:
            raise NotImplementedError

        # Bounding box proposal network and classification network are shared
        self.loc_head = self._make_head(self.num_anchors * 4, num_features)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes,
                num_features)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous() \
                    .view(x.size(0), -1, 4)
            # [N,9*90,H,W] -> [N,H,W,9*90] -> [N,H*W*9,90]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous() \
                    .view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _make_head(self, out_planes, num_features):
        layers = []
        for _ in range(4):
            layers.append(conv3x3(num_features, num_features, bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(num_features, out_planes, bias=True))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2, 3, 224, 224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

# test()
