from __future__ import print_function

import os
import argparse

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser(description='Test Retinanet')
parser.add_arument('--model', default='ckpt.pth', type=str,
        help='saved checkpoint in folder \"checkpoint\"')
parser.add_arument('--image', default='sample.jpg', type=str,
        help='path to input image in folder \"image\"')
parser.add_arument('--output', default='out.bmp', type=str,
        help='path to result image in folder \"image\"')
args = parser.parse_args()

print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('./checkpoint/' + args.model)['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),

    # Normalize with ImageNet statistics
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('./image/' + args.image)
w = h = 600
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()

print('Saving result..')
img.save('./image/' + args.output)
