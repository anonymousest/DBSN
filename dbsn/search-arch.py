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

from torch.utils.data import DataLoader, Subset
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli

import os
import sys
import math
import numpy as np
np.set_printoptions(precision=4)

import shutil
import glob
import logging

import stochastic_nn as stochastic_nn
from architect import _ada_gumbel_softmax, _logp_ada_concrete
import matplotlib.pyplot as plt


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
    parser.add_argument('--nEpochs', type=int, default=20)
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='svhn')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--tau_min', type=float, default=1., help='tau_min')
    parser.add_argument('--net_learning_rate', type=float, default=1e-2, help='learning rate for net')
    parser.add_argument('--net_weight_decay', type=float, default=1e-4, help='wd for arch encoding')
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

    parser.add_argument('--restore', type=str, default=None)
    args = parser.parse_args()

    args.save = '../work/sa{}'.format(args.restore.split('/')[-2].replace('run', ''))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'), mode="w+")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    if args.dataset == "cifar10":
        normTransform = transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
        nClasses = 10
        dataset_train = dset.CIFAR10(root='../cifar', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normTransform
                                     ]))
        dataset_test = dset.CIFAR10(root='../cifar', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normTransform
                                    ]))
    elif args.dataset=="cifar100":
        normTransform = transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        nClasses = 100
        dataset_train = dset.CIFAR100(root='../cifar', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normTransform
                                      ]))
        dataset_test = dset.CIFAR100(root='../cifar', train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         normTransform
                                     ]))
    elif args.dataset=="svhn":
        normTransform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        nClasses = 10
        dataset_train = dset.SVHN(root='../svhn', split='train', download=True,
                                  transform=transforms.Compose([
                                      # transforms.RandomCrop(32, padding=4),
                                      # transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normTransform
                                  ]))
        dataset_test = dset.SVHN(root='../svhn', split='test', download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     normTransform
                                 ]))

    trainLoader = DataLoader(
                      dataset_train,
                      batch_size=args.batchSz, shuffle=True,
                      pin_memory=True, num_workers=2, drop_last=True)

    testLoader = DataLoader(
                     dataset_test,
                     batch_size=args.batchSz, shuffle=False,
                     pin_memory=True, num_workers=2)

    num_batch = len(trainLoader)
    logging.info(str(num_batch))

    if args.restore:
        net = torch.load(args.restore)
        if isinstance(net,torch.nn.DataParallel):
            net = net.module
        alphas_org = torch.load(args.restore.replace("epoch", "alphas"))
        betas_org = torch.load(args.restore.replace("epoch", "betas"))
        alphas = Variable(1e-3*torch.randn(int((args.nlayers-1) * args.nlayers / 2), 4).cuda(), requires_grad=True)
        betas = Variable(torch.ones(int((args.nlayers-1) * args.nlayers / 2), 1).cuda()*math.log(1.), requires_grad=True)

        logging.info('  + Number of params of trained model: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))

        arch_optimizer = torch.optim.Adam([alphas, betas],
                lr=args.arch_learning_rate, betas=(args.beta1, 0.999))

    else:
        exit(1)

    print(torch.cat([F.softmax(alphas, 1), betas.exp()], 1).data.cpu().numpy())
    test(args, 10, testLoader, net, alphas_org, betas_org)
    test(args, 10, testLoader, net, alphas, betas)
    alphas_prior_logits = (torch.ones_like(alphas) / float(alphas.size()[1])).log()
    K = alphas.size()[1]
    kl_weight = 1/float(num_batch)

    for epoch in range(1, args.nEpochs + 1):
        net.train()

        nProcessed = 0
        nTrain = len(trainLoader.dataset)
        train_loss = 0
        incorrect = 0

        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = data.cuda(), target.cuda()

            arch_optimizer.zero_grad()
            alphas_instances, alphas_instances_log = _ada_gumbel_softmax(alphas, args.tau_min, betas, True)
            kls = _logp_ada_concrete(alphas, betas, args.tau_min, alphas_instances_log, K).sum() - \
                       _logp_ada_concrete(alphas_prior_logits, torch.cuda.FloatTensor([0.]), args.tau_min, alphas_instances_log, K).sum()

            output = net(data, alphas_instances)
            losses = F.nll_loss(output, target, reduction='sum') + kls * kl_weight
            loss = losses / args.batchSz
            loss.backward()
            arch_optimizer.step()

            nProcessed += len(data)
            train_loss += loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()
        train_loss /= len(trainLoader) # loss function already averages over batch size
        err = 100.*incorrect/nProcessed
        logging.info('Epoch: {}, training arch, average loss: {:.4f}, Error: {}/{} ({:.0f}%)'.format(epoch, train_loss, incorrect, nProcessed, err))
        if epoch % 10 == 0:
            test(args, 10, testLoader, net, alphas, betas)
            print(torch.cat([F.softmax(alphas, 1), betas.exp()], 1).data.cpu().numpy())
            if epoch % 20 == 0:
                torch.save(net, os.path.join(args.save, 'epoch{}.pth'.format(epoch)))
                torch.save(alphas, os.path.join(args.save, 'alphas{}.pth'.format(epoch)))
                torch.save(betas, os.path.join(args.save, 'betas{}.pth'.format(epoch)))


def test(args, ites, testLoader, net, alphas, betas):
    test_loss = 0
    incorrect = 0
    net.eval()

    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        outputs = []
        with torch.no_grad():
            for ite in range(ites):
                output = net(data, _ada_gumbel_softmax(alphas, args.tau_min, betas))
                outputs.append(output)
        output = torch.logsumexp(torch.stack(outputs, 0), 0) - np.log(float(ites))
        test_loss += F.nll_loss(output, target).item()*(data.size()[0])
        pred = output.data.max(1)[1]
        incorrect += pred.ne(target.data).cpu().sum()

    nTotal = len(testLoader.dataset)
    test_loss /= nTotal
    err = 100.*float(incorrect)/float(nTotal)

    logging.info('Test set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%)'.format(
        test_loss, incorrect, nTotal, err))

if __name__=='__main__':
    main()
