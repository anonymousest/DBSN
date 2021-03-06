"""
    100-layer tiramisu/fc densenet67 model definition
"""

import torch
import torch.nn as nn
from torchvision import transforms

from .layers import DenseBlock, TransitionDown, TransitionUp, Bottleneck
from .sto_layers import StoSequential, StochasticBlock
from .joint_transforms import JointRandomResizedCrop, JointRandomHorizontalFlip, JointCompose, LabelToLongTensor, JointRandomVerticalFlip

__all__ = ['FCDenseNet57', 'FCDenseNet67', 'FCDenseNet103', 'DBSN']

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, num_classes=11):
        super().__init__()

        self.num_classes = num_classes

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=num_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)

        return out


class FCDenseNet57:
    base = FCDenseNet
    args = list()
    kwargs = {'in_channels':3, 'down_blocks':(4, 4, 4, 4, 4),
            'up_blocks':(4, 4, 4, 4, 4), 'bottleneck_layers':4,
            'growth_rate':12, 'out_chans_first_conv':48}

    camvid_mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    camvid_std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])
    transform_test =  transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])

    joint_transform = JointCompose([
        JointRandomResizedCrop(224), # commented for fine-tuning
        JointRandomHorizontalFlip()
    ])
    ft_joint_transform = JointCompose([
        JointRandomHorizontalFlip()
    ])
    target_transform = transforms.Compose([
        LabelToLongTensor(),
    ])

class FCDenseNet67:
    base = FCDenseNet
    args = list()

    kwargs = {'in_channels':3, 'down_blocks':(5, 5, 5, 5, 5),
            'up_blocks':(5, 5, 5, 5, 5), 'bottleneck_layers':5,
            'growth_rate':16, 'out_chans_first_conv':48}

    camvid_mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    camvid_std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    transform_train = transforms.Compose([
          transforms.ToTensor(),
          # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])
    transform_test =  transforms.Compose([
          transforms.ToTensor(),
          # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])

    joint_transform = JointCompose([
        JointRandomResizedCrop(224), # commented for fine-tuning
        JointRandomHorizontalFlip()
    ])
    ft_joint_transform = JointCompose([
        JointRandomHorizontalFlip()
    ])

    target_transform = transforms.Compose([
        LabelToLongTensor(),
    ])

class FCDenseNet103:
    base = FCDenseNet
    args = list()

    kwargs = {'in_channels':3, 'down_blocks':(4,5,7,10,12),
            'up_blocks':(12,10,7,5,4), 'bottleneck_layers':15,
            'growth_rate':16, 'out_chans_first_conv':48}

    camvid_mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    camvid_std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    transform_train = transforms.Compose([
          transforms.ToTensor(),
          #transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])
    transform_test =  transforms.Compose([
          transforms.ToTensor(),
          #transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])

    joint_transform = JointCompose([
        JointRandomResizedCrop(224), # commented for fine-tuning
        JointRandomHorizontalFlip()
    ])
    ft_joint_transform = JointCompose([
        JointRandomHorizontalFlip()
    ])

    target_transform = transforms.Compose([
        LabelToLongTensor(),
    ])



class StochasticNet(nn.Module):
    def __init__(self, in_channels=3, nlayers_per_cell=7, ncells_per_block=3, ncells_bottleneck=1,
                 ndown_blocks=5, nup_blocks=5,
                 growth_rate=16, out_chans_first_conv=48, num_classes=11, after_norm_type='bn'):
        super().__init__()

        self.num_classes = num_classes

        self.nlayers_per_cell = nlayers_per_cell
        self.ncells_per_block = ncells_per_block
        self.ndown_blocks = ndown_blocks
        self.nup_blocks = nup_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=False))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(ndown_blocks):
            tmp = []
            for _ in range(ncells_per_block):
                tmp.append(StochasticBlock(cur_channels_count, growth_rate, nlayers_per_cell, after_norm_type))
                cur_channels_count += (nlayers_per_cell*growth_rate)

            self.denseBlocksDown.append(StoSequential(*tmp))
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count, False, False))

        #####################
        #     Bottleneck    #
        #####################

        tmp = []
        for _ in range(ncells_bottleneck):
            tmp.append(StochasticBlock(cur_channels_count, growth_rate, nlayers_per_cell, after_norm_type))
            cur_channels_count += (nlayers_per_cell*growth_rate)

        self.add_module('bottleneck',StoSequential(*tmp))
        prev_block_channels = growth_rate*nlayers_per_cell*ncells_bottleneck

        # self.add_module('bottleneck',Bottleneck(cur_channels_count,
        #                              growth_rate, 5, False, False))
        # prev_block_channels = growth_rate*5
        # cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(nup_blocks):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels, False))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            tmp = []
            for _ in range(ncells_per_block):
                tmp.append(StochasticBlock(cur_channels_count, growth_rate, nlayers_per_cell, after_norm_type))
                cur_channels_count += (nlayers_per_cell*growth_rate)
            self.denseBlocksUp.append(StoSequential(*tmp))
            prev_block_channels = growth_rate*nlayers_per_cell*ncells_per_block

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=num_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)

    def forward(self, x, alpha=None):
        alphas = [None, None] if alpha is None else alpha.chunk(2, dim=0)
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.denseBlocksDown)):
            out = self.denseBlocksDown[i](out, alphas[0])
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        org_channels = out.size()[1]
        out = self.bottleneck(out, alphas[0])[:, org_channels:, :, :]
        for i in range(len(self.denseBlocksUp)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            if i < len(self.denseBlocksUp) - 1:
                org_channels = out.size()[1]
                out = self.denseBlocksUp[i](out, alphas[1])[:, org_channels:, :, :]
            else:
                out = self.denseBlocksUp[i](out, alphas[1])

        out = self.finalConv(out)

        return out

class DBSN:
    base = StochasticNet
    args = list()

    kwargs = {'in_channels':3, 'nlayers_per_cell': 5, 'ncells_per_block':1, 'ncells_bottleneck':1,
            'ndown_blocks':5, 'nup_blocks':5,
            'growth_rate':16, 'out_chans_first_conv':48, 'after_norm_type': 'bn'}

    camvid_mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    camvid_std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

    transform_train = transforms.Compose([
          transforms.ToTensor(),
          # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])
    transform_test =  transforms.Compose([
          transforms.ToTensor(),
          # transforms.Normalize(mean=camvid_mean, std=camvid_std),
    ])

    joint_transform = JointCompose([
        JointRandomResizedCrop(224), # commented for fine-tuning
        JointRandomHorizontalFlip()
    ])
    ft_joint_transform = JointCompose([
        JointRandomHorizontalFlip()
    ])

    target_transform = transforms.Compose([
        LabelToLongTensor(),
    ])
