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
  'bn' : lambda C, g : nn.BatchNorm2d(C, affine=False, track_running_stats=True),
  'in' : lambda C, g : nn.InstanceNorm2d(C),
  'gn' : lambda C, g : nn.GroupNorm(g, C, affine=False),
  'none' : lambda C, g : NO_NORM(),
}

def drop_path(x, drop_prob, g, k):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), g, 1, 1, 1).bernoulli_(keep_prob)).repeat(1, 1, k, 1, 1).view(x.size(0), g*k, 1, 1)
    x.div_(keep_prob)
    x.mul_(mask)
  return x

class BNReLUConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1, affine=True, after_norm_type="bn"):
    super(BNReLUConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      NORMS[after_norm_type](C_out, groups)
    )

  def forward(self, x):
    return self.op(x)

class StdConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, groups=1, affine=True, after_norm_type="bn"):
    super(StdConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine),
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
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
      NORMS[after_norm_type](C_out, groups)
      )

  def forward(self, x):
    return self.op(x)

class SepDilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, affine=True, after_norm_type="bn"):
    super(SepDilConv, self).__init__()
    self.op = nn.Sequential(
      nn.BatchNorm2d(C_in, affine=affine),
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
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False, groups=groups),
      nn.BatchNorm2d(C_in, affine=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False, groups=groups),
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

class Layer(nn.Module):
    def __init__(self, growthRate, idx, drop_rate, after_norm_type, droppath_rate):
        super(Layer, self).__init__()
        self.growthRate = growthRate
        self.idx = idx
        self.drop_rate = drop_rate
        self.droppath_rate = droppath_rate

        self._ops = nn.ModuleList()
        self._ops.append(Zero())
        self._ops.append(Identity(growthRate*idx, idx, after_norm_type=after_norm_type))
        # self._ops.append(StdConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))
        # self._ops.append(DilConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))
        self._ops.append(SepConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))
        self._ops.append(SepDilConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type))
        # self._ops.append(SepConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type, kernel_size=5, padding=2))
        # self._ops.append(SepDilConv(growthRate*idx, growthRate*idx, groups=idx, after_norm_type=after_norm_type, kernel_size=5, padding=4, dilation=2))

    def forward(self, x, feature_i, row_alphas):
        if not row_alphas is None:
            row_alphas_expand = row_alphas[:, None, :].repeat(1, self.growthRate, 1).view(self.idx*self.growthRate, 4)
            out = sum(row_alphas_expand[:, j][None, :, None, None] * op(x) for j, op in enumerate(self._ops))
        else:
            out = sum([self._ops[0](x) / 4., self._ops[1](x) / 4.,
                       drop_path(self._ops[2](x), self.droppath_rate, self.idx, self.growthRate) / 4., drop_path(self._ops[3](x), self.droppath_rate, self.idx, self.growthRate) / 4.
                      ])

        out_size = out.size()
        out = (out.view(out_size[0], self.idx, self.growthRate, out_size[2], out_size[3]).sum(1) + feature_i)# / (self.idx + 1)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=True)

        out = torch.cat((x, out), 1)
        return out

class StochasticBlock(nn.Module):
    def __init__(self, nChannels, growthRate, nLayers, drop_rate, after_norm_type, droppath_rate):
        super(StochasticBlock, self).__init__()

        self.growthRate = growthRate
        self.drop_rate = drop_rate
        self.droppath_rate = droppath_rate
        self.nLayers = nLayers
        self.preprocess = BNReLUConv(nChannels, growthRate * nLayers, 1, 1, 0, affine=True, groups=nLayers, after_norm_type=after_norm_type)

        layers = []
        for i in range(1, int(nLayers)):
            layers.append(Layer(growthRate, i, drop_rate, after_norm_type, droppath_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, input, alphas):
        features0 = self.preprocess(input).split(self.growthRate, 1)
        output = features0[0]
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=True)

        for i, module in enumerate(self.layers):
            output = module(output, features0[i+1], None if alphas is None
                            else alphas[int(i*(i+1)/2): int((i+1)*(i+2)/2), :])
        return torch.cat((input, output), 1)

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, drop_rate=0.0):
        super(Transition, self).__init__()
        self.drop_rate = drop_rate
        self.op = nn.Sequential(
            nn.BatchNorm2d(nChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        out = self.op(x)
        return out

class StoSequential(nn.Sequential):
    def __init__(self, *args):
        super(StoSequential, self).__init__(*args)

    def forward(self, input, alpha):
        for module in self._modules.values():
            input = module(input, alpha)
        return input

class StochasticNet(nn.Module):
    def __init__(self, growthRate, reduction, nClasses, args):
        super(StochasticNet, self).__init__()

        nLayers = args.nlayers

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)

        tmp = []
        for _ in range(args.ncells):
            tmp.append(StochasticBlock(nChannels, growthRate, nLayers, args.drop_rate, args.after_norm_type, args.droppath_rate))
            nChannels += nLayers*growthRate
        self.sto1 = StoSequential(*tmp)

        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels, args.drop_rate)
        nChannels = nOutChannels

        #growthRate *= 2
        tmp = []
        for _ in range(args.ncells):
            tmp.append(StochasticBlock(nChannels, growthRate, nLayers, args.drop_rate, args.after_norm_type, args.droppath_rate))
            nChannels += nLayers*growthRate
        self.sto2 = StoSequential(*tmp)

        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels, args.drop_rate)
        nChannels = nOutChannels

        #growthRate *= 2
        tmp = []
        for _ in range(args.ncells):
            tmp.append(StochasticBlock(nChannels, growthRate, nLayers, args.drop_rate, args.after_norm_type, args.droppath_rate))
            nChannels += nLayers*growthRate
        self.sto3 = StoSequential(*tmp)

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if args.init_type == 'origin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is None and m.bias is None:
                    continue
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, alphas_instance=None, return_logits=False):
        out = self.conv1(x)
        out = self.trans1(self.sto1(out, alphas_instance))
        out = self.trans2(self.sto2(out, alphas_instance))
        out = self.sto3(out, alphas_instance)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        logits = self.fc(out)
        logp = F.log_softmax(logits, -1)
        if return_logits:
            return logits, logp
        else:
            return logp
