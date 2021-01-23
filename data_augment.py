import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import shutil
import argparse

transforms_perspective = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomPerspective(distortion_scale=0.6, p=1),
    transforms.ToTensor(),
    ])

dataset = torchvision.datasets.ImageFolder(root=os.getcwd() + '/data/', transform=transforms_perspective)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
