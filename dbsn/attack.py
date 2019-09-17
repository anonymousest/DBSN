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
from architect import _ada_gumbel_softmax
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
    parser.add_argument('--batchSz', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tau0', type=float, default=3., help='tau0')
    parser.add_argument('--tau_min', type=float, default=1., help='tau_min')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--method', type=str, default='dbsn')
    parser.add_argument('--ps', action='store_true', default=False, help='point estimation')
    parser.add_argument('--drop_rate', type=float, default=0., help='drop_rate')
    parser.add_argument('--droppath_rate', type=float, default=0., help='droppath_rate')

    parser.add_argument('--adv_method', type=str, default='fgsm')
    parser.add_argument('--epsilon', type=float, default=-1, help='epsilon')
    args = parser.parse_args()

    args.save = '../'#'../work/attack{}'.format(args.restore.split('/')[-2].replace('run', ''))
    #create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log_attack_' + args.adv_method + '.txt'), mode="w+")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    testTransform = transforms.Compose([
        transforms.ToTensor(),
    ])
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    if args.dataset == 'cifar10':
        normMean = torch.cuda.FloatTensor([0.49139968, 0.48215827, 0.44653124]).view(1, 3, 1, 1)
        normStd = torch.cuda.FloatTensor([0.24703233, 0.24348505, 0.26158768]).view(1, 3, 1, 1)

        testLoader = DataLoader(
            dset.CIFAR10(root='../../../cifar', train=False, download=True, transform=testTransform),
            batch_size=args.batchSz, shuffle=False, **kwargs)

    elif args.dataset == 'cifar100':
        normMean = torch.cuda.FloatTensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
        normStd = torch.cuda.FloatTensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)

        testLoader = DataLoader(
            dset.CIFAR100(root='../../../cifar', train=False, download=True, transform=testTransform),
            batch_size=args.batchSz, shuffle=False, **kwargs)

    if args.restore:
        net = torch.load(args.restore)
        alphas = torch.load(args.restore.replace("epoch", "alphas"))
        betas = torch.load(args.restore.replace("epoch", "betas"))

        alphas_prior_logits = (torch.ones_like(alphas) / float(alphas.size()[1])).log()

        if isinstance(net,torch.nn.DataParallel) and torch.cuda.device_count() == 1:
            net = net.module
        net.eval()

        logging.info('  + Number of params: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))
    else:
        exit(1)


    ites = 1 if (args.method == 'densenet' or (args.method == 'dbsn' and args.ps)) and args.drop_rate == 0. and args.droppath_rate == 0. else 30
    logging.info('<<<<' + str(ites) + '>>>>')
    if args.epsilon == -1:
        for eps in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
            attack(args, ites, net, testLoader, alphas_prior_logits, alphas, betas, normMean, normStd, eps)
    else:
        attack(args, ites, net, testLoader, alphas_prior_logits, alphas, betas, normMean, normStd, args.epsilon)

def get_grad(args, ites, data, target, net, alphas_prior_logits, alphas, betas, normMean, normStd):
    grads = []

    for ite in range(ites):
        if args.method == 'dbsn':
            if args.ps:
                output = net((data - normMean) / normStd, F.softmax(alphas, 1))
            else:
                output = net((data - normMean) / normStd, _ada_gumbel_softmax(alphas, args.tau_min, betas))
        elif args.method == 'densenet':
            output = net((data - normMean) / normStd)
        elif args.method == 'random':
            output = net((data - normMean) / normStd, _ada_gumbel_softmax(alphas_prior_logits, args.tau_min))
        loss = F.nll_loss(output, target)
        grad = torch.autograd.grad(
            [loss], [data], grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
        grads.append(grad.detach())
    return sum(grads) / float(ites)

def attack(args, ites, net, testLoader, alphas_prior_logits, alphas, betas, normMean, normStd, epsilon):

    if args.adv_method == "fgsm":
        alpha = epsilon #8. / 255
        iteration = 1
    elif args.adv_method == "bim":
        alpha = epsilon / 3. #2. / 255
        iteration = 3
    if epsilon == 0.:
        iteration = 0

    test_loss = 0
    incorrect = 0
    entropy = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        image_max = torch.clamp(data + epsilon, 0., 1.)
        image_min = torch.clamp(data - epsilon, 0., 1.)
        adv_data = data

        for _ in range(iteration):
            adv_data.requires_grad_()
            grad = get_grad(args, ites, adv_data, target, net, alphas_prior_logits, alphas, betas, normMean, normStd)
            adv_data = adv_data + alpha * torch.sign(grad)
            adv_data = torch.max(torch.min(adv_data, image_max), image_min).detach()

        data_n = (adv_data - normMean) / normStd
        outputs = []
        with torch.no_grad():
            for ite in range(ites):
                if args.method == 'dbsn':
                    if args.ps:
                        output = net(data_n, F.softmax(alphas, 1))
                    else:
                        output = net(data_n, _ada_gumbel_softmax(alphas, args.tau_min, betas))
                elif args.method == 'densenet':
                    output = net(data_n)
                elif args.method == 'random':
                    output = net(data_n, _ada_gumbel_softmax(alphas_prior_logits, args.tau_min))
                outputs.append(output)

        output = torch.logsumexp(torch.stack(outputs, 0), 0) - np.log(float(ites))
        entropy += (- output * output.exp()).sum()
        test_loss += F.nll_loss(output, target).item()*(data.size()[0])
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    nTotal = len(testLoader.dataset)
    test_loss /= nTotal
    entropy /= nTotal
    err = 100.*float(incorrect)/float(nTotal)

    logging.info('Test set: epsilon: {} average loss: {:.4f}, entropy: {:.4f}, Error: {}/{} ({:.2f}%)'.format(
        epsilon, test_loss, entropy, incorrect, nTotal, err))

if __name__=='__main__':
    main()
