from datahandler import *
from autoaugment import CIFAR10Policy, Cutout
from thop import clever_format, profile

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

# Added
import time
import sys
from scalable_senet import *
from conv_mix import *

import argparse

#--lr-max=0.05 --ra-n=2 --ra-m=12 --wd=0.005 --scale=1.0 --jitter=0 --reprob=0
parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="ConvMixer")
parser.add_argument('--dataLoc', type=str, default="./data")

parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--scale', default=0.75, type=float)
parser.add_argument('--reprob', default=0.25, type=float)
parser.add_argument('--ra-m', default=8, type=int)
parser.add_argument('--ra-n', default=1, type=int)
parser.add_argument('--jitter', default=0.1, type=float)

parser.add_argument('--hdim', default=64, type=int)
parser.add_argument('--depth', default=8, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)

parser.add_argument('--wd', default=0.001, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--lr-max', default=0.142, type=float)
parser.add_argument('--workers', default=1, type=int)

args = parser.parse_args()


def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
    return acc


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total

    return acc


torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformations

initial_image_size = 32
total_classes = 10
number_input_channels = 3

print('==> Preparing data..')
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(initial_image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Architecture
print('==> Building network architecture..')
# model = scaled_senet(1, 0.67, initial_image_size)
model = ConvMixer(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=10)
model = nn.DataParallel(model).cuda()

model.to(device)
print(model)

if device == 'cuda':
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Optimizer
print('==> Defining the Optimizer and its hyperparameters..')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9 , weight_decay= 0.001,dampening= 0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=240, eta_min=1e-8)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=140, eta_min=0.01)


# if opt_flag=1:
   # optimizer = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd)
# else:
# optimizer = optim.SGD(model.parameters(), lr=0.042, momentum=0.9, weight_decay=0.005)
# 
# scaler = torch.cuda.amp.GradScaler()




# --------------------------------------------
# Dataset - Cifar10
# Plugin new dataset here
# --------------------------------------------

trainset = torchvision.datasets.CIFAR10(
    root=args.dataLoc, train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root=args.dataLoc, train=False, download=True, transform=transform_test)

y_train = trainset.targets
y_test = testset.targets

subset_indices_1, subset_indices_test_1 = get_subset_data(y_train, y_test)
partial_trainset = torch.utils.data.Subset(trainset, subset_indices_1)
partial_testset = torch.utils.data.Subset(testset, subset_indices_test_1)

# --------------------------------------------
# End of the dataset portion
# we need partial_trainset and partial_testset to define the trainloader and testloader
# --------------------------------------------

print('==> Model initialization..')
initialize(model)

trainloader = torch.utils.data.DataLoader(
    partial_trainset, batch_size=512, num_workers=2, shuffle=True)

testloader = torch.utils.data.DataLoader(
    partial_testset, batch_size=128, shuffle=False)

start_epoch = 0
training_accuracies = []
testing_accuracies = []
t0 = time.time()
execution_time = 0
total_epochs = 0
epoch = 0
best_test_acc = 0

while execution_time < 600:
    tr_acc = train(epoch)
    training_accuracies.append(tr_acc)
    te_acc = test(epoch)
    testing_accuracies.append(te_acc)
    if epoch <= 260:
        scheduler.step()
    execution_time = time.time() - t0

    if te_acc > best_test_acc:
        best_test_acc = te_acc
        print('Saving checkpoint..')
        state = {
            'net': model.state_dict(),
            'acc': best_test_acc,
            'epoch': epoch,
        }
        torch.save(state, 'ckpt.pth')
    lr = scheduler.get_last_lr()[0]

    print(
        "Epoch {}, Execution time: {:.1f}, LR: {:.3f}, Train accuracy: {:.3f}, Val accuracy: {:.3f} "
            .format(epoch, execution_time, lr, tr_acc, best_test_acc))

    epoch += 1

print('Best valid acc', max(testing_accuracies))
print('Best train acc', max(training_accuracies))
