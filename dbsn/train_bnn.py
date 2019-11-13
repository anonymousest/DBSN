#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


import os
import sys
import math
import numpy as np
np.set_printoptions(precision=4)

import shutil
import glob
import logging

from torchbnn import stochastic_nn
from torchbnn.architect import PSOptimizer, GumbelSoftmaxMOptimizer, _ada_gumbel_softmax

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--run', type=str, default='bnn_dbsn_0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cifar10')

    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--net_learning_rate', type=float, default=0.1, help='learning rate for net')
    parser.add_argument('--net_weight_decay', type=float, default=0., help='wd for arch encoding')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--init_type', type=str, default='origin')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--nesterov', action='store_true', default=False, help='use nesterov for net')
    parser.add_argument('--ncells', type=int, default=4)
    parser.add_argument('--nlayers', type=int, default=7)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--drop_rate', type=float, default=0., help='drop_rate')
    parser.add_argument('--droppath_rate', type=float, default=0., help='droppath_rate')
    parser.add_argument('--after_norm_type', type=str, default='bn')

    parser.add_argument('--tau0', type=float, default=3., help='tau0')
    parser.add_argument('--tau_min', type=float, default=1., help='tau_min')
    parser.add_argument('--tau_anneal_rate', type=float, default=0.000015, help='tau_anneal_rate')

    parser.add_argument('--ps', action='store_true', default=False, help='point estimation')
    parser.add_argument('--prior_sigma', type=float, default=0.1, help='prior_sigma')

    parser.add_argument('--restore_arch', action='store_true', default=False, help='restore_arch')
    parser.add_argument('--w_kl_weight', type=float, default=0.001, help='w_kl_weight')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = '../work/run{}'.format(args.run)
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'), mode="w+")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(str(args))

    np.random.seed(args.seed)
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    ngpus = int(torch.cuda.device_count())
    logging.info("# of GPUs: " + str(ngpus))

    if args.dataset == "cifar10":
        normMean = [0.49139968, 0.48215827, 0.44653124]
        normStd = [0.24703233, 0.24348505, 0.26158768]
        normTransform = transforms.Normalize(normMean, normStd)
        nClasses = 10
        dataset_all = dset.CIFAR10
    elif args.dataset=="cifar100":
        normMean = [0.5071, 0.4867, 0.4408]
        normStd = [0.2675, 0.2565, 0.2761]
        normTransform = transforms.Normalize(normMean, normStd)
        nClasses = 100
        dataset_all = dset.CIFAR100

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    trainLoader = DataLoader(
        dataset_all(root='../cifar', train=True, download=True,
                    transform=trainTransform),
                    batch_size=args.batchSz, shuffle=True,
                    pin_memory=True, num_workers=2, drop_last=True)
    trainLoader_arch = None
    testLoader = DataLoader(
        dataset_all(root='../cifar', train=False, download=True,
                     transform=testTransform),
                     batch_size=args.batchSz, shuffle=False,
                     pin_memory=True, num_workers=2)

    num_batch = len(trainLoader)
    logging.info(str(num_batch))

    start_epoch = 1
    net = stochastic_nn.StochasticNet(growthRate=16, reduction=0.4, nClasses=nClasses, args=args).cuda()

    restore_alpha_path = "../work/runadags_lr3_decayto0.5_0/alphas100.pth" if args.dataset=='cifar10' else "../work/runadags_lr3_d205_cifar100_0/alphas100.pth"
    alphas = torch.load(restore_alpha_path) if args.restore_arch else Variable(1e-3*torch.randn(int((args.nlayers-1) * args.nlayers / 2), 4).cuda(), requires_grad=True)
    betas = Variable(torch.ones(int((args.nlayers-1) * args.nlayers / 2), 1).cuda()*math.log(1.), requires_grad=True)

    logging.info('  + Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.net_learning_rate, nesterov=args.nesterov,
                            momentum=0.9, weight_decay=args.net_weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    if args.ps:
        biOptimizer = PSOptimizer(net, args, num_batch, alphas, betas, ngpus)
    else:
        biOptimizer = GumbelSoftmaxMOptimizer(net, args, num_batch, alphas, betas, ngpus)

    for epoch in range(start_epoch, args.nEpochs + 1):

        adjust_opt(args.opt, optimizer, biOptimizer, epoch, args.arch_learning_rate, args.nEpochs)
        train_arch(args, epoch, net, trainLoader, trainLoader_arch, optimizer, biOptimizer, ngpus)
        if epoch % 10 == 1:
            print(torch.cat([F.softmax(alphas, 1), betas.exp()], 1).data.cpu().numpy())
        if epoch % 20 == 0:
            test(args, epoch, num_batch, net, testLoader, biOptimizer, alphas, betas, ngpus)
            torch.save(net, os.path.join(args.save, 'epoch{}.pth'.format(epoch)))
            torch.save(alphas, os.path.join(args.save, 'alphas{}.pth'.format(epoch)))
            torch.save(betas, os.path.join(args.save, 'betas{}.pth'.format(epoch)))

def train_arch(args, epoch, net, trainLoader, trainLoader_arch, optimizer, biOptimizer, ngpus):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    train_loss = 0
    incorrect = 0
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda().repeat(ngpus, 1,1,1), target.cuda().repeat(ngpus)
        data_search, target_search = None, None
        output, loss = biOptimizer.step(batch_idx, data, target, data_search, target_search, optimizer, epoch)
        nProcessed += len(data)
        train_loss += loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()
    train_loss /= len(trainLoader) # loss function already averages over batch size
    err = 100.*incorrect/nProcessed
    logging.info('Epoch: {}, training arch, average loss: {:.4f}, Error: {}/{} ({:.0f}%)'.format(epoch, train_loss, incorrect, nProcessed, err))

def test(args, epoch, num_batch, net_p, testLoader, biOptimizer, alphas, betas, ngpus):

    if isinstance(net_p, nn.DataParallel):
        net = net_p.module
    else:
        net = net_p
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()
        outputs = []
        with torch.no_grad():
            for ite in range(100):
                if args.ps:
                    output = net(data, F.softmax(alphas, 1))
                else:
                    output = net(data, _ada_gumbel_softmax(alphas, biOptimizer.tau, betas))
                outputs.append(output)
        output = torch.logsumexp(torch.stack(outputs, 0), 0) - np.log(float(len(outputs)))
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    logging.info('Test set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)'.format(
        test_loss, incorrect, nTotal, err))

def adjust_opt(optAlg, optimizer, biOptimizer, epoch, arch_lr0, nEpochs):
    if optAlg == 'sgd':
        if epoch == 1:
            lr = 1e-1
            arch_lr = arch_lr0
        elif epoch == 61:
            lr = 1e-2
            arch_lr = arch_lr0 * 0.3
        elif epoch == 81:
            lr = 1e-3
            arch_lr = arch_lr0 * 0.1
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # for param_group in biOptimizer.logit_optim.param_groups:
        #     param_group['lr'] = arch_lr


if __name__=='__main__':
    main()
