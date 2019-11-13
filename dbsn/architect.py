import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, binary_cross_entropy, softmax, log_softmax, kl_div, sigmoid
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import clamp_probs
from numpy.random import logistic

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
    def __init__(self, model, args, num_batch, alphas, betas, ngpus):
      self.model = model
      self.alphas = alphas
      self.betas = betas
      self.ngpus = ngpus

      # self.betas.requires_grad = True
      # self.betas.grad = None
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
        self.logit_optim.zero_grad()
        network_optimizer.zero_grad()

        alphas_instances, kls = [], []
        for _ in range(self.ngpus):
            tmp1, tmp2 = _ada_gumbel_softmax(self.alphas, self.tau, self.betas, True)
            alphas_instances.append(tmp1)
            kls.append(_logp_ada_concrete(self.alphas, self.betas, self.tau, tmp2, self.K).sum() -
                       _logp_ada_concrete(self.alphas_prior_logits, torch.cuda.FloatTensor([0.]), self.tau, tmp2, self.K).sum()
                      )

        # output = self.model(input_train, torch.cat(alphas_instances[:2], 0))
        # output1 = self.model(input_train, torch.cat(alphas_instances[2:], 0))
        # losses = torch.cat([F.nll_loss(output, target_train, reduction='none').view(2, -1).sum(1), F.nll_loss(output1, target_train, reduction='none').view(2, -1).sum(1)], 0) + torch.stack(kls) * self.kl_weight

        output = self.model(input_train, torch.cat(alphas_instances, 0))
        losses = F.nll_loss(output, target_train, reduction='none').view(self.ngpus, -1).sum(1) + torch.stack(kls) * self.kl_weight

        # if epoch <= 60:
        #     temp = -0.1 if self.args.dataset == "cifar10" else -0.1
        # elif epoch <= 80:
        #     temp = -1 if self.args.dataset == "cifar10" else -0.3
        # else:
        #     temp = -10 if self.args.dataset == "cifar10" else -0.9
        temp = 1
        weights = 1./self.ngpus# if self.args.avg else softmax(-losses*temp, 0).detach()
        loss = (losses * weights).sum() / self.args.batchSz
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.logit_optim.step()
        network_optimizer.step()

        self.counter += 1

        return output, loss


class PSOptimizer(object):
    def __init__(self, model, args, num_batch, alphas, betas, ngpus):
      self.model = model
      self.alphas = alphas
      self.ngpus = ngpus
      self.logit_optim = torch.optim.Adam([self.alphas],
          lr=args.arch_learning_rate, betas=(args.beta1, 0.999), weight_decay=1e-3)

      self.args = args
      self.counter = 0
      self.num_batch = num_batch
      self.tau = None


    def step(self, batch_idx, input_train, target_train, input_search, target_search, network_optimizer, epoch):

        self.logit_optim.zero_grad()
        network_optimizer.zero_grad()

        output = self.model(input_train, F.softmax(self.alphas, 1))
        loss = F.nll_loss(output, target_train)
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.logit_optim.step()
        network_optimizer.step()

        self.counter += 1

        return output, loss

class DARTSOptimizer(object):
    def __init__(self, model, args, num_batch, alphas, betas, ngpus):
      self.model = model
      self.alphas = alphas
      self.ngpus = ngpus
      self.logit_optim = torch.optim.Adam([self.alphas],
          lr=args.arch_learning_rate, betas=(args.beta1, 0.999), weight_decay=1e-3)

      self.args = args
      self.counter = 0
      self.num_batch = num_batch


    def step(self, batch_idx, input_train, target_train, input_search, target_search, network_optimizer, epoch):

        network_optimizer.zero_grad()
        output = self.model(input_train, F.softmax(self.alphas, 1).detach())
        loss = F.nll_loss(output, target_train)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        network_optimizer.step()

        self.logit_optim.zero_grad()
        output1 = self.model(input_search, F.softmax(self.alphas, 1))
        loss1 = F.nll_loss(output1, target_search)
        loss1.backward()
        self.logit_optim.step()

        self.counter += 1

        return output, loss
