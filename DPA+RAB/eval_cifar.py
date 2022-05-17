from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
sys.path.append('./FeatureLearningRotNet/architectures')

from NetworkInNetwork import NetworkInNetwork
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy
import random
import re
from attack_lib import attack_DPA 

torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache() 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(description='PyTorch CIFAR Certification')
parser.add_argument('--models',  type=str, help='name of models directory')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'

if not os.path.exists('./evaluations'):
    os.makedirs('./evaluations')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data.. ', args.models)

modelnames = list(map(lambda x: './checkpoints/'+args.models+'/'+x, list(filter( lambda x:x[0]!='.',os.listdir('./checkpoints/'+args.models)))))
num_classes = 10
predictions = torch.zeros(10000, len(modelnames),num_classes).cuda()
labels = torch.zeros(10000).type(torch.int).cuda()
firstit = True


for i in range(len(modelnames)):
    modelname = modelnames[i]
    seed = int(re.findall(r"partition_.*\.pth",  modelname)[-1][10:-4])
    if (args.zero_seed):
        seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from cifar10_cnn_model import Model
    net = Model(gpu=True) 

    checkpoint = torch.load(modelname)
    
    # print("Checkpoint Network: ", checkpoint['net'])
    net.load_state_dict(checkpoint['net'])
    
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # [poison_set, bening_set, testloader, BATCH_SIZE, N_EPOCH, LR] = attack_DPA(testset, testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=1)

    net.eval()
    batch_offset = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = net(inputs)
            predictions[batch_offset:inputs.size(0)+batch_offset,i,:] = out
            ous = out.argmax(1)
            # print("Out max: ", ous, " Shape: ", ous.size())
            # print("Target: ", targets, " shape: ", targets.size()) 
            print("Acc: ", 100 * (ous == targets).sum().item()/1024)
            if firstit:
                labels[batch_offset:batch_offset+inputs.size(0)] = targets
            batch_offset += inputs.size(0) 
    firstit = False 
    torch.cuda.empty_cache()
print('labels: ', labels, 'scores: ', predictions)
torch.save({'labels': labels, 'scores': predictions},'./evaluations/'+args.models+'.pth')
