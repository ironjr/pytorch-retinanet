from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable

from logger import Logger

# Debug
from math import isnan


# Datasets are downloaded from http://cocodataset.org/
data_root = '../common/datasets/'
coco17_train_path = 'COCO/train2017/'
coco17_val_path = 'COCO/val2017/'

save_every = 5000

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--label', default='default', type=str, help='run label')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
start_epoch = 0  # start from epoch 0 or last epoch
iteration = 0

# Tensorboard logger
log_root = './log/'
log_names = ['', 'loc_loss', 'cls_loss', 'train_loss', 'avg_train_loss', 'test_loss']
log_dirs = list(map(lambda x: log_root + args.label + '/' + x, log_names))
for log_dir in log_dirs:
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
_, loc_logger, cls_logger, train_logger, avg_train_logger, test_logger = list(map(lambda x: Logger(x), log_dirs))

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),

    # Normalize with ImageNet statistics
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

# Train with COCO
trainset = ListDataset(root=(data_root + coco17_train_path),
        list_file='./data/coco17_train.txt', train=True, transform=transform,
        input_size=600)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=trainset.collate_fn)

testset = ListDataset(root=(data_root + coco17_val_path),
        list_file='./data/coco17_val.txt', train=False, transform=transform,
        input_size=600)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False,
        num_workers=0, collate_fn=testset.collate_fn)

# Model
net = RetinaNet()

criterion = FocalLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

checkpoint = None
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
#      optimizer.load_state_dict(checkpoint['optim'])
    start_epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    print("Last loss: %.3f" % (checkpoint['loss']))
    print("Training start from epoch %d iteration %d" % (start_epoch, iteration))
else:
    net.load_state_dict(torch.load('./model/net.pth'))

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        # Offset correction
        idx = batch_idx + iteration

        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss, loc_loss, cls_loss = criterion(
                loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data

        step = idx + len(trainloader) * epoch
        if loc_logger is not None:
            loc_logger.scalar_summary('loss', loc_loss.data, step)
        if cls_logger is not None:
            cls_logger.scalar_summary('loss', cls_loss.data, step)
        if train_logger is not None:
            train_logger.scalar_summary('loss', loss.data, step)
        if avg_train_logger is not None:
            avg_train_logger.scalar_summary('loss', train_loss / (batch_idx + 1), step)
        print('batch (%d/%d) | loc_loss: %.3f | cls_loss: %.3f | train_loss: %.3f | avg_loss: %.3f' 
                % (idx, len(trainloader), loc_loss.data, cls_loss.data, loss.data, train_loss / (batch_idx + 1)))

        # Save at the middle
        if batch_idx % save_every == 0:
            save('train' + str(step), net, optimizer,
                    train_loss / (batch_idx + 1), epoch, idx + 1)

        # Finish when total iterations match the number of batches
        if (idx + 1) % len(trainloader) == 0:
            break

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss, loc_loss, cls_loss = criterion(
                loc_preds, loc_targets, cls_preds, cls_targets)

        test_loss += loss.data

        print('batch (%d/%d) | loc_loss: %.3f | cls_loss: %.3f | test_loss: %.3f | avg_loss: %.3f' 
                % (batch_idx, len(testloader), loc_loss.data, cls_loss.data, loss.data, test_loss / (batch_idx + 1)))

    # Save and log
    test_loss /= len(testloader)
    save('test' + str(epoch), net, optimizer, test_loss, epoch + 1)

    if test_logger is not None:
        test_logger.scalar_summary('loss', test_loss, (epoch + 1) * len(trainloader))
    print('average test loss: %.3f' % (test_loss))


# Save checkpoints
def save(label, net, optimizer, loss=float('inf'), epoch=0, iteration=0):
    print('==> Saving checkpoint')
    state = {
        'net': net.module.state_dict(),
        'optim': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'iteration': iteration,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_' + label + '.pth')
    print('==> Save done!')


#  for epoch in range(start_epoch, start_epoch + 200):
#      train(epoch)
#      test(epoch)
train(start_epoch)
test(start_epoch)
