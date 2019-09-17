import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class NO_NORM(nn.Module):

  def __init__(self,):
    super(NO_NORM, self).__init__()

  def forward(self, x):
    return x

NORMS = {
  'bn' : lambda C, g : nn.BatchNorm2d(C, affine=False, track_running_stats=False),
  'in' : lambda C, g : nn.InstanceNorm2d(C),
  'gn' : lambda C, g : nn.GroupNorm(g, C, affine=False),
  'none' : lambda C, g : NO_NORM(),
  'ln': lambda C, g: nn.GroupNorm(1, C, affine=False)
}

class BNReLUConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1, affine=True, after_norm_type="bn"):
    super(BNReLUConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine, track_running_stats=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      NORMS[after_norm_type](C_out, groups)
    )

  def forward(self, x):
    return self.op(x)

class SepDilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, affine=True, after_norm_type="bn"):
    super(SepDilConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine, track_running_stats=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False, groups=groups),
      NORMS[after_norm_type](C_out, groups)
      )

  def forward(self, x):
    return self.op(x)

class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, groups=1, affine=True, after_norm_type="bn"):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine, track_running_stats=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False, groups=groups),
      nn.BatchNorm2d(C_in, affine=True, track_running_stats=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False, groups=groups),
      NORMS[after_norm_type](C_out, groups)
      )

  def forward(self, x):
    return self.op(x)

class StdConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, groups=1, affine=True, after_norm_type="bn"):
    super(StdConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine, track_running_stats=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
      NORMS[after_norm_type](C_out, groups)
      )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, affine=True, after_norm_type="bn"):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine, track_running_stats=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
      NORMS[after_norm_type](C_out, groups)
      )

  def forward(self, x):
    return self.op(x)

class Identity(nn.Module):

  def __init__(self, C, groups=1, after_norm_type="bn"):
    super(Identity, self).__init__()
    self.op = NORMS[after_norm_type](C, groups)

  def forward(self, x):
    return self.op(x)


class Zero(nn.Module):

  def __init__(self):
    super(Zero, self).__init__()

  def forward(self, x):
    return 0. #x.mul(0.)

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

class Layer(nn.Module):
    def __init__(self, growthRate, idx, after_norm_type):
        super(Layer, self).__init__()
        self.growthRate = growthRate
        self.idx = idx

        self._ops = nn.ModuleList()
        #self._ops.append(Zero())
        self._ops.append(Identity(growthRate*idx, idx, after_norm_type=after_norm_type))
        self._ops.append(SepConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))
        self._ops.append(SepDilConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))
        # self._ops.append(StdConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))
        # self._ops.append(DilConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))

        # self._ops.append(SepConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type, kernel_size=5, padding=2))
        # self._ops.append(SepDilConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type, kernel_size=5, padding=4, dilation=2))

    def forward(self, x, feature_i, row_alphas):
        if not row_alphas is None:
            row_alphas_expand = row_alphas[:, None, :].repeat(1, self.growthRate, 1).view(self.idx*self.growthRate, 3)
            out = sum(row_alphas_expand[:, j][None, :, None, None] * op(x) for j, op in enumerate(self._ops))
        else:
            out = sum(op(x) for op in self._ops) / 3.

        out_size = out.size()
        out = out.view(out_size[0], self.idx, self.growthRate, out_size[2], out_size[3]).sum(1) + drop_path(feature_i, 0. if self.training else 0.)

        # if self.drop_rate > 0:
        #     out = F.dropout(out, p=self.drop_rate, training=True)

        out = torch.cat((x, out), 1)
        return out

class StochasticBlock(nn.Module):
    def __init__(self, nChannels, growthRate, nLayers, after_norm_type):
        super(StochasticBlock, self).__init__()

        self.growthRate = growthRate
        self.nLayers = nLayers
        self.preprocess = BNReLUConv(nChannels, growthRate * nLayers, 3, 1, 1, affine=True, groups=nLayers, after_norm_type=after_norm_type)

        layers = []
        for i in range(1, int(nLayers)):
            layers.append(Layer(growthRate, i, after_norm_type))
        self.layers = nn.ModuleList(layers)

        print('Number of params of stoblock: {}'.format(sum([p.data.nelement() for p in self.parameters()])))
        # print('Number of params of preprocess: {}'.format(sum([p.data.nelement() for p in self.preprocess.parameters()])))
        # for ly in self.layers:
        #     print('Number of params of std and dil: {}, {}'.format(sum([p.data.nelement() for p in ly._ops[2].parameters()]), sum([p.data.nelement() for p in ly._ops[3].parameters()])))
        #
        # exit(1)

    def forward(self, input, alphas):
        features0 = self.preprocess(input).split(self.growthRate, 1)
        output = features0[0]
        # if self.drop_rate > 0:
        #     output = F.dropout(output, p=self.drop_rate, training=True)
        assert(alphas.size()[0] == int((self.nLayers-1)*(self.nLayers)/2))
        for i, module in enumerate(self.layers):
            output = module(output, features0[i+1], None if alphas is None else alphas[int(i*(i+1)/2): int((i+1)*(i+2)/2), :])
        return torch.cat((input, output), 1)

class StoSequential(nn.Sequential):
    def __init__(self, *args):
        super(StoSequential, self).__init__(*args)

    def forward(self, input, alpha):
        for module in self._modules.values():
            input = module(input, alpha)
        return input
