# -*- coding: utf-8 -*-
import model as m
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import random
import math
import os
import random
from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd
import dataloader as d
from torch.utils.tensorboard import SummaryWriter
import sys

def val(model, device, test_loader, epoch, writer):
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    loss_vec = []
    with torch.no_grad():
        for data, target in test_loader:
            targets = target.cpu().numpy()
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            criterion = nn.CrossEntropyLoss()
            loss_vec.append(criterion(output, target).item())
    loss_vec = np.array(loss_vec)     
    print("Validation Accuracy is: "+str(100. * correct / len(test_loader.dataset)))
    writer.add_scalar('Validation Acc', correct / len(test_loader.dataset), epoch)
    writer.add_scalar('Validation Loss', np.mean(loss_vec), epoch)
    

def train(model, device, train_loader, optimizer, epoch, writer):
    train_acc = 0
    train_loss = 0
    loss_vec = []
    for it,(local_batch, local_labels) in enumerate(train_loader):
        batch = torch.tensor(local_batch, requires_grad=True).cuda()
        labels = local_labels.cuda()
        optimizer.zero_grad()
        out = model(batch)
        _, predicted = torch.max(out, 1)
        total = labels.shape[0]
        train_acc += (predicted == labels).sum().item()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        loss_vec.append(loss.item())
        loss.backward()
        optimizer.step()
    loss_vec = np.array(loss_vec)
    print("#####EPOCH "+str(epoch)+"#####")
    print("Train accuracy is: "+str(train_acc / len(train_loader.dataset)*100.))
    print("Train loss is: "+str(np.mean(loss_vec)))
    writer.add_scalar('Train Acc', train_acc / len(train_loader.dataset), epoch)
    writer.add_scalar('Train Loss', np.mean(loss_vec), epoch)

        
def main():
    #argparse settings
    parser = argparse.ArgumentParser(description='PyTorch MSTAR Example') #400 and 0.001
    parser.add_argument('--train-batchsize', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 400)')
    parser.add_argument('--test-batchsize', type=int, default=400, metavar='N',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.015, metavar='LR',
                        help='learning rate (default: 0.015)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Adam momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default="/home/saschaho/roof_data/", metavar='N',
                        help='where data is stored')
    parser.add_argument('--use-pretrain', type=int, default=0, metavar='N',
                        help='Use pretrained model or not')
    parser.add_argument('--checkpoint-dir', type=str, default='./ckpt', metavar='N',
                        help='checkpoint directory')
    parser.add_argument('--model-name', type=str, default='resnet18', metavar='N',
                        help='model to try from torch models')
    parser.add_argument('--multi-gpu', type=int, default=0, metavar='N',
                        help='Whether using multiple gpus')
    
    args = parser.parse_args()
    ckpt_dir = args.checkpoint_dir
    os.system('rm -r '+ckpt_dir)
    os.makedirs(ckpt_dir+'/runs', exist_ok=True)
    
    ## Tensorboard ##
    writer = SummaryWriter(ckpt_dir+'/runs')
    writer.add_text('args', str(sys.argv), 0)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = m.Model(args.model_name).cuda()
        
    if args.use_pretrain:
        model.load_state_dict(torch.load('./pretrained_model.ckpt'))
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#Model Parameters: "+str(params))
    
    # Multiple GPUs for larger networks such as resnet50
    if args.multi_gpu:
        model = nn.DataParallel(model)
    
    ### For local use of this code, please change the directory of the images to your own path! ###
    train_loader, test_loader = d.get_dataloader(data_dir=args.data_dir, train_batch=args.train_batchsize, test_batch=args.test_batchsize)
    print("Training batch Size: "+str(args.train_batchsize))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, amsgrad=True)
    print("Learning Rate: "+str(args.lr))
    
    for epoch in range(1, args.epochs + 1):
        val(model, device, test_loader, epoch, writer)
        train(model, device, train_loader, optimizer, epoch, writer)

if __name__ == '__main__':
    main()

