import os
import sys
import math
import string
import random
import shutil

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils

from torch.nn.functional import relu, binary_cross_entropy, softmax, log_softmax, kl_div, sigmoid
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import clamp_probs
from numpy.random import logistic

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'


def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return round(err,5)

def train(model, trn_loader, optimizer, criterion, biOptimizer, epoch, ngpus, dbsn=False):
    model.train()
    trn_loss = 0
    trn_error = 0
    inputs_list, targets_list = [], []
    for idx, (inputs, targets) in enumerate(trn_loader):
        inputs = inputs.cuda(non_blocking = True)
        targets = targets.cuda(non_blocking = True)

        if ngpus > 1:
            inputs_list.append(inputs)
            targets_list.append(targets)

            if len(inputs_list) < ngpus:
                continue

        if dbsn:
            tmp_targets = torch.cat(targets_list, 0) if ngpus > 1 else targets

            output, loss = biOptimizer.step(idx, torch.cat(inputs_list, 0) if ngpus > 1 else inputs, tmp_targets, optimizer, epoch, criterion, ngpus)
            # optimizer.zero_grad()
            # output = model(torch.cat(inputs_list, 0) if ngpus > 1 else inputs)
            # loss = criterion(output, tmp_targets)
            # loss.backward()
            # # nn.utils.clip_grad_norm_(model.parameters(), 5.)
            # optimizer.step()

            inputs_list, targets_list = [], []
            _, _, trn_acc_curr = numpy_metrics(output.data.cpu().numpy(), tmp_targets.data.cpu().numpy())

        else:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            _, _, trn_acc_curr = numpy_metrics(output.data.cpu().numpy(), targets.data.cpu().numpy())

        trn_loss += loss.item() * ngpus
        trn_error += (1 - trn_acc_curr) * ngpus

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, alphas, betas, biOptimizer, ngpus, dbsn=False, num_classes = 11, ntimes=5, mcdropout=False, dir=None):
    # if isinstance(model_p, nn.DataParallel):
    #     model = model_p.module
    # else:
    #     model = model_p
    if not mcdropout:
        model.eval()

    save_img, save_lab, save_pred = [], [], []
    with torch.no_grad():
        test_loss = 0
        test_error = 0
        I_tot = np.zeros(num_classes)
        U_tot = np.zeros(num_classes)

        inputs_list, targets_list = [], []
        counter = 0
        for data, target in test_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if ngpus > 1:
                inputs_list.append(data)
                targets_list.append(target)

                if len(inputs_list) < ngpus:
                    continue

            if dbsn:
                outputs = 0
                for ite in range(ntimes):
                    output = model(data if ngpus == 1 else torch.cat(inputs_list, 0), torch.cat([_ada_gumbel_softmax(alphas, biOptimizer.tau, betas) for _ in range(ngpus)], 0))
                    outputs += F.softmax(output, 1)
                output = outputs.log() - np.log(float(ntimes))
                #print(output.exp().sum(1)[0])
            else:
                outputs = 0
                for ite in range(ntimes if mcdropout else 1):
                    output = model(data if ngpus == 1 else torch.cat(inputs_list, 0))
                    outputs += F.softmax(output, 1)
                output = outputs.log() - np.log(float(ntimes if mcdropout else 1))

            loss = criterion(output, target if ngpus == 1 else torch.cat(targets_list, 0))
            test_loss += loss * ngpus

            # compute current mIOU statistic
            I, U, acc = numpy_metrics(output.cpu().numpy(), target.cpu().numpy() if ngpus == 1 else torch.cat(targets_list, 0).cpu().numpy(), n_classes=num_classes, void_labels=[num_classes])
            inputs_list, targets_list = [], []
            I_tot += I
            U_tot += U
            test_error += (1 - acc) * ngpus

            if dir:
                save_img.append(data.cpu().numpy() if ngpus == 1 else torch.cat(inputs_list, 0).cpu().numpy())
                save_lab.append(target.cpu().numpy() if ngpus == 1 else torch.cat(targets_list, 0).cpu().numpy())
                save_pred.append(output.cpu().numpy())
            counter += 1

        if dir:
            np.savez('/home/' + dir + "_pred.npz", imgs=np.concatenate(save_img, 0), labs=np.concatenate(save_lab, 0), preds=np.concatenate(save_pred, 0))
        test_loss /= len(test_loader)
        test_error /= len(test_loader)
        print(I_tot / U_tot)
        m_jacc = np.mean(I_tot / U_tot)

        return test_loss.item(), test_error, m_jacc

def numpy_metrics(y_pred, y_true, n_classes = 11, void_labels=[11]):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    from: https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    void label is 11 by default
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=1)

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~ np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def weights_init(m, init_type="kaiming_normal"):
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_uniform(m.weight)
    #     m.bias.data.zero_()

    if isinstance(m, nn.Conv2d):
        if init_type == 'origin':
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        else:
            nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is None and m.bias is None:
            #print(m)
            return
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    inputs, label = next(iter(loader))
    data = inputs.cuda()
    #label = label.cuda()
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)

    # normalize inputs if necessary
    img_means = inputs.mean((1,2)).unsqueeze(1).unsqueeze(2)
    img_std = torch.tensor([inputs[:,i,:,:].std() for i in range(3)]).unsqueeze(1).unsqueeze(2)

    inputs = (inputs - img_means) / img_std

    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(label[i])
        img_utils.view_annotated(pred[i])

def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)

def masked_ce_loss(y_pred, y_true, void_class = 11., weight=None, reduce = True, return_mask=False):
    """
    y_pred: output predictded probabilities
    Y_true: true labels
    void_class: background/void class label
    weight: if weights are use
    reduce: if reduction is appleid to final loss

    masked version of crossentropy loss
    ported over from the masked version in
    https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    """

    el = torch.ones_like(y_true) * void_class
    mask = torch.ne(y_true, el).long()

    y_true_tmp = y_true * mask

    loss = F.cross_entropy(y_pred, y_true_tmp, weight=weight, reduction='none')
    loss = mask.float() * loss

    if reduce:
        if return_mask:
            return loss.sum()/mask.sum(), mask.sum()
        else:
            return loss.sum()/mask.sum()
    else:
        return loss, mask

def schedule(epoch, lr_init, epochs):
    """
    piecewise learning rate schedule
    borrowed from https://github.com/timgaripov/swa
    """
    t = (epoch) / (epochs)
    lr_ratio = 0.01
    if epoch <= 350:
        factor = 1.0
    elif epoch <= 750:
        factor = 1.0 - (1.0 - lr_ratio) * (epoch - 350) / 400
    else:
        factor = lr_ratio*(-0.009*epoch + 7.75) #(9.1 - 9.*t) #add here
    return lr_init * factor

    # if epoch <= 300:
    #     factor = 1.0
    # elif epoch <= 600:
    #     factor = 0.1 #1.0 - (1.0 - lr_ratio) * (epoch - 350) / 400
    # else:
    #     factor = 0.01 #lr_ratio*(-0.009*epoch + 7.75) #(9.1 - 9.*t) #add here
    # return lr_init * factor

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

      # self.betas.requires_grad = True
      # self.betas.grad = None
      self.logit_optim = torch.optim.Adam([self.alphas],#, self.betas],
          lr=args.arch_learning_rate, betas=(0.5, 0.999))
      self.betas.requires_grad = False

      self.args = args

      self.counter = 0
      self.num_batch = num_batch

      self.K = self.alphas.size()[-1]
      self.alphas_prior = torch.ones_like(self.alphas) / float(self.K)
      self.alphas_prior_logits = self.alphas_prior.log()
      self.betas_prior = None
      self.kl_weight = 1/float(self.num_batch)

    def step(self, batch_idx, input_train, target_train, network_optimizer, epoch, criterion, ngpus):

        self.tau = np.maximum(self.args.tau0 * np.exp(-self.args.tau_anneal_rate * self.counter), self.args.tau_min)
        if self.counter % 1000 == 0:
            print("ite: ", self.counter, self.tau)
        self.betas.fill_(math.log(max(1.001-0.001*epoch, 0.5)))
        self.logit_optim.zero_grad()
        network_optimizer.zero_grad()

        alphas_instances, kls = [], []
        for _ in range(ngpus):
            tmp1, tmp2 = _ada_gumbel_softmax(self.alphas, self.tau, self.betas, True)
            alphas_instances.append(tmp1)
            kls.append(_logp_ada_concrete(self.alphas, self.betas, self.tau, tmp2, self.K).sum() -
                       _logp_ada_concrete(self.alphas_prior_logits, torch.cuda.FloatTensor([0.]), self.tau, tmp2, self.K).sum()
                      )
        output = self.model(input_train, torch.cat(alphas_instances, 0))
        losses, masks = criterion(output, target_train, reduce=False)
        losses = (losses.view(ngpus, -1, target_train.size()[1], target_train.size()[2]).sum([1,2,3]) + torch.stack(kls) * self.kl_weight) / masks.view(ngpus, -1, target_train.size()[1], target_train.size()[2]).sum([1,2,3]).float()
        loss = (losses / ngpus).sum()


        # alphas_instances, alphas_instances_log = _ada_gumbel_softmax(self.alphas, self.tau, self.betas, True)
        # kls = _logp_ada_concrete(self.alphas, self.betas, self.tau, alphas_instances_log, self.K).sum() - \
        #            _logp_ada_concrete(self.alphas_prior_logits, torch.cuda.FloatTensor([0.]), self.tau, alphas_instances_log, self.K).sum()
        #
        # output = self.model(input_train, alphas_instances)
        # loss, mask_sum = criterion(output, target_train, return_mask=True)
        # loss += kls * self.kl_weight / mask_sum
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        self.logit_optim.step()
        network_optimizer.step()

        self.counter += ngpus
        return output, loss
