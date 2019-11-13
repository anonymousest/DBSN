#!/usr/bin/env python3

import argparse
import torch
import torchbnn

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

from architect import _ada_gumbel_softmax
import matplotlib.pyplot as plt

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        bin_boundaries_plot = torch.linspace(0, 1, 11)
        self.bin_lowers_plot = bin_boundaries_plot[:-1]
        self.bin_uppers_plot = bin_boundaries_plot[1:]

    def forward(self, logits, labels, title):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        accuracy_in_bin_list = []
        for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            accuracy_in_bin = 0
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean().item()
            accuracy_in_bin_list.append(accuracy_in_bin)

        p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align = 'edge', edgecolor ='black')
        p2 = plt.plot([0,1], [0,1], '--', color='gray')

        plt.ylabel('Accuracy')
        plt.xlabel('Confidence')
        #plt.title(title)
        plt.xticks(np.arange(0, 1.01, 0.2))
        plt.yticks(np.arange(0, 1.01, 0.2))
        plt.xlim(left=0,right=1)
        plt.ylim(bottom=0,top=1)
        plt.grid(True)
        #plt.legend((p1[0], p2[0]), ('Men', 'Women'))

        plt.savefig(title + "/cal.pdf")


        return ece

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
    parser.add_argument('--batchSz', type=int, default=800)
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
    parser.add_argument('--ent', action='store_true', default=False, help='only calulate entropy')
    parser.add_argument('--test_mc_eff', action='store_true', default=False, help='the number of mc')
    parser.add_argument('--test_bnn', action='store_true', default=False, help='if test with weight uncertainty')
    args = parser.parse_args()

    args.save = '../'#'../work/attack{}'.format(args.restore.split('/')[-2].replace('run', ''))
    #create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log_test.txt'))
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

    svhn_testloader = torch.utils.data.DataLoader(
        dset.SVHN(root='../../../svhn', split='test', download=True, transform=testTransform),
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

    if args.test_bnn:
        ites = 100
    else:
        ites = 1 if (args.method == 'densenet' or (args.method == 'dbsn' and args.ps)) and args.drop_rate == 0. and args.droppath_rate == 0. else 100
    print('<<<<' + str(ites) + '>>>>')
    if args.ent:
        test(args, ites, net, svhn_testloader, alphas_prior_logits, alphas, betas, normMean, normStd, True)
    else:
        if args.test_mc_eff:
            errs, eces, test_losses = [], [], []
            ites_list = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,60,70,80,90,100]
            for ites in ites_list:
                r1, r2, r3 = test(args, ites, net, testLoader, alphas_prior_logits, alphas, betas, normMean, normStd, False)
                print(r1, r2, r3)
                errs.append(r1)
                eces.append(r2.item())
                test_losses.append(r3)
            print(errs, eces, test_losses)

            fig = plt.figure()
            # plt.subplot(1,2,1)
            plt.plot(ites_list, errs, color='c', lw=2)
            plt.xlabel('Number of MC samples', fontsize=14)
            plt.ylabel('Error rate (%)', color='k', fontsize=14)
            plt.tick_params('y', colors='k')
            # plt.grid(True)
            plt.savefig(args.save + "/mc_times_errs_" + args.dataset + ".pdf")

            fig = plt.figure()
            # plt.subplot(1,2,2)
            plt.plot(ites_list, eces, color='y', lw=2)
            plt.xlabel('Number of MC samples', fontsize=14)
            plt.ylabel('ECE', color='k', fontsize=14)
            plt.tick_params('y', colors='k')
            # plt.grid(True)
            plt.savefig(args.save + "/mc_times_eces_" + args.dataset + ".pdf")

            fig = plt.figure()
            # plt.subplot(1,2,1)
            plt.plot(ites_list, test_losses, color='g', lw=2)
            plt.xlabel('Number of MC samples', fontsize=14)
            plt.ylabel('Test loss', color='k', fontsize=14)
            plt.tick_params('y', colors='k')
            # plt.grid(True)
            plt.savefig(args.save + "/mc_times_tls_" + args.dataset + ".pdf")

        else:
            test(args, ites, net, testLoader, alphas_prior_logits, alphas, betas, normMean, normStd, False)


def test(args, ites, net, testLoader, alphas_prior_logits, alphas, betas, normMean, normStd, ent):
    test_loss = 0
    incorrect = 0

    entropies = []
    logits = []
    labels = []
    for data, target in testLoader:

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data_n = (data - normMean) / normStd

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
        logits.append(output)
        labels.append(target)
        test_loss += F.nll_loss(output, target).item()*(data.size()[0])
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()
        entropies.append((- output * output.exp()).sum(1))

    if ent:
        entropies = torch.cat(entropies, 0).data.cpu().numpy()
        np.save(args.save + "/svhn_entropies.npy", entropies)
        return

    nTotal = len(testLoader.dataset)
    test_loss /= nTotal
    err = 100.*float(incorrect)/float(nTotal)
    ece = _ECELoss()(torch.cat(logits, 0), torch.cat(labels, 0), args.save)

    logging.info('Test set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%, Ece loss: {:.4f})'.format(
        test_loss, incorrect, nTotal, err, ece.item()))
    entropies = torch.cat(entropies, 0).data.cpu().numpy()
    np.save(args.save + "/entropies.npy", entropies)

    return err, ece, test_loss

if __name__=='__main__':
    main()
