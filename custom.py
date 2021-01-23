import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--name', default='best', type=str, help='name of the file')
parser.add_argument('--epoch', default=100, type=int, help='num of epoch')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
transform_custom = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_custom = torchvision.datasets.ImageFolder(root=os.getcwd() + '/custom/', transform=transform_custom)
customloader = torch.utils.data.DataLoader(dataset_custom, batch_size=1, shuffle=True, num_workers=2)

print('==> Building model..')

net = models.resnet34()
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net = net.to(device)

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/' + args.name + '.pth')
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(customloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            print(F.softmax(outputs))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(predicted.detach().cpu().numpy()[0])
            if predicted.detach().cpu().numpy()[0] == 1:
                print('It it predicted as : Gingivitis, Accuracy of prediction : {}'.format(F.softmax(outputs).detach()
                                                                                            .cpu().numpy()[0][1]))
            else:
                print('It it predicted as : Normal, Accuracy of prediction : {}'.format((F.softmax(outputs).detach()
                    .cpu().numpy()[0][0])))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


for epoch in range(start_epoch, start_epoch + args.epoch):
    test(epoch)
    scheduler.step()

