#!/usr/bin/env python3

import argparse
import torch
import torchsso
from torchsso.optim import SecondOrderOptimizer, VIOptimizer, VOGN

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

import os
import sys
import math
import numpy as np
np.set_printoptions(precision=4)

import shutil
import glob
import logging

import stochastic_nn as stochastic_nn
from test import _ECELoss

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

def _ada_gumbel_softmax(logits, tau, scale=None, return_logits=False, eps=1e-10):
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    if scale is not None:
        gumbels *= torch.exp(scale)
    gumbels = (logits + gumbels) / tau
    if return_logits:
        return gumbels.softmax(-1), gumbels
    else:
        return gumbels.softmax(-1)

def _logp_ada_concrete(logits, scale, tau, value, K = 4):
    score = (logits - value.mul(tau)) / torch.exp(scale)
    score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
    return score - (K - 1) * scale

class GumbelSoftmaxMOptimizer(object):
    def __init__(self, model, args, num_batch, alphas, betas):
      self.model = model
      self.alphas = alphas
      self.betas = betas
      self.logit_optim = torch.optim.Adam([self.alphas],#, self.betas],
          lr=args.arch_learning_rate, betas=(args.beta1, 0.999))
      self.betas.requires_grad = False
      self.args = args
      self.counter = 0
      self.num_batch = num_batch
      self.K = self.alphas.size()[-1]
      self.alphas_prior = torch.ones_like(self.alphas) / float(self.K)
      self.alphas_prior_logits = self.alphas_prior.log()
      self.betas_prior = None
      self.kl_weight = 1/float(self.num_batch)

    def step(self, batch_idx, input_train, target_train, input_search, target_search, network_optimizer, epoch):

        self.tau = np.maximum(self.args.tau0 * np.exp(-self.args.tau_anneal_rate * self.counter), self.args.tau_min)
        self.betas.fill_(math.log(max(1.01-0.01*epoch, 0.5)))

        def closure():
            self.logit_optim.zero_grad()
            network_optimizer.zero_grad()
            alphas_instances, tmp2 = _ada_gumbel_softmax(self.alphas, self.tau, self.betas, True)
            kl = _logp_ada_concrete(self.alphas, self.betas, self.tau, tmp2, self.K).sum() - \
                  _logp_ada_concrete(self.alphas_prior_logits, torch.cuda.FloatTensor([0.]), self.tau, tmp2, self.K).sum()
            output = self.model(input_train, alphas_instances)
            loss = F.nll_loss(output, target_train) + kl*self.kl_weight/self.args.batchSz
            loss.backward()
            return loss, output
        loss, output = network_optimizer.step(closure=closure, arch_optim=self.logit_optim)
        self.counter += 1
        return output, loss


class PSOptimizer(object):
    def __init__(self, model, args, num_batch, alphas, betas):
      self.model = model
      self.alphas = alphas
      self.logit_optim = torch.optim.Adam([self.alphas],
          lr=args.arch_learning_rate, betas=(args.beta1, 0.999), weight_decay=1e-3)
      self.args = args
      self.counter = 0
      self.num_batch = num_batch
      self.tau = None


    def step(self, batch_idx, input_train, target_train, input_search, target_search, network_optimizer, epoch):
        def closure():
            self.logit_optim.zero_grad()
            network_optimizer.zero_grad()
            output = self.model(input_train, F.softmax(self.alphas, 1))
            loss = F.nll_loss(output, target_train)
            loss.backward()
            return loss, output

        loss, output = network_optimizer.step(closure=closure, arch_optim=self.logit_optim)
        self.counter += 1
        return output, loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=32)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--run', type=str, default='bnn_torchsso_0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cifar10')

    parser.add_argument('--net_learning_rate', type=float, default=0.01, help='learning rate for net')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--tau0', type=float, default=3., help='tau0')
    parser.add_argument('--tau_min', type=float, default=1., help='tau_min')
    parser.add_argument('--tau_anneal_rate', type=float, default=0.000015, help='tau_anneal_rate')
    parser.add_argument('--init_type', type=str, default='origin')
    parser.add_argument('--ncells', type=int, default=1)
    parser.add_argument('--nlayers', type=int, default=4)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--drop_rate', type=float, default=0., help='drop_rate')
    parser.add_argument('--droppath_rate', type=float, default=0., help='droppath_rate')

    parser.add_argument('--ps', action='store_true', default=False, help='point estimation')
    parser.add_argument('--after_norm_type', type=str, default='bn')

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
    alphas = Variable(1e-3*torch.randn(int((args.nlayers-1) * args.nlayers / 2), 4).cuda(), requires_grad=True)
    betas = Variable(torch.ones(int((args.nlayers-1) * args.nlayers / 2), 1).cuda()*math.log(1.), requires_grad=True)

    logging.info('  + Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    optim_kwargs = {"curv_type": "Cov", "lr": args.net_learning_rate, "grad_ema_decay": 0.1,
                    "grad_ema_type": "raw", "num_mc_samples": 1, "val_num_mc_samples": 100,
                    "warmup_kl_weighting_init": 0.01, "warmup_kl_weighting_steps": 1000,
                    "init_precision": 8e-3, "prior_variance": 1, "acc_steps": 1,
                    "curv_shapes": {
                      "Conv2d": "Diag",
                      "Linear": "Diag",
                      "BatchNorm1d": "Diag",
                      "BatchNorm2d": "Diag"
                    }}
    curv_args = {"damping": 1e-7, "ema_decay": 0.01}
    # optimizer = VOGN(net, dataset_size=len(trainLoader.dataset))
    optimizer =  VIOptimizer(net, dataset_size=len(trainLoader.dataset), seed=args.seed,
                             **optim_kwargs, curv_kwargs=curv_args)

    if args.ps:
        biOptimizer = PSOptimizer(net, args, num_batch, alphas, betas)
    else:
        biOptimizer = GumbelSoftmaxMOptimizer(net, args, num_batch, alphas, betas)

    for epoch in range(start_epoch, args.nEpochs + 1):
        train_arch(args, epoch, net, trainLoader, trainLoader_arch, optimizer, biOptimizer)
        if epoch % 10 == 1:
            print(torch.cat([F.softmax(alphas, 1), betas.exp()], 1).data.cpu().numpy())
        if epoch % 100 == 0:
            test(args, epoch, num_batch, net, optimizer, testLoader, biOptimizer, alphas, betas)
            # torch.save(net, os.path.join(args.save, 'epoch{}.pth'.format(epoch)))
            # torch.save(alphas, os.path.join(args.save, 'alphas{}.pth'.format(epoch)))
            # torch.save(betas, os.path.join(args.save, 'betas{}.pth'.format(epoch)))
            # torch.save(optimizer, os.path.join(args.save, 'optimizer{}.pth'.format(epoch)))

def train_arch(args, epoch, net, trainLoader, trainLoader_arch, optimizer, biOptimizer):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    train_loss = 0
    incorrect = 0
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()
        output, loss = biOptimizer.step(batch_idx, data, target, None, None, optimizer, epoch)
        nProcessed += len(data)
        train_loss += loss
        pred = output.data.max(1)[1]
        incorrect += pred.ne(target.data).cpu().sum()
    train_loss /= len(trainLoader)
    err = 100.*incorrect/nProcessed
    logging.info('Epoch: {}, training arch, average loss: {:.4f}, Error: {}/{} ({:.0f}%)'.format(epoch, train_loss, incorrect, nProcessed, err))

def test(args, epoch, num_batch, net, optimizer, testLoader, biOptimizer, alphas, betas):

    net.eval()
    test_loss = 0
    incorrect = 0
    logits = []
    labels = []
    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            if args.ps:
                output = optimizer.prediction(data, alphas, biOptimizer.tau, betas, F.softmax)
            else:
                output = optimizer.prediction(data, alphas, biOptimizer.tau, betas, _ada_gumbel_softmax)
        logits.append(output)
        labels.append(target)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()


    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    ece = _ECELoss()(torch.cat(logits, 0), torch.cat(labels, 0), args.save)
    logging.info('Test set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%), ECE: {}'.format(
        test_loss, incorrect, nTotal, err, ece))

if __name__=='__main__':
    main()
