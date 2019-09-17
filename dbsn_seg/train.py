"""
    training script for segmentation models
    partial port of our own train/run_swag.py file
    note: no options to train swag-diag
"""

import time
from pathlib import Path
import numpy as np
import os, sys, math, glob
import argparse, shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from functools import partial

import utils.training as train_utils
from datasets.data import camvid_loaders
import models
from torch.autograd import Variable
import logging

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

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--dataset', type=str, default='CamVid')
parser.add_argument('--data_path', type=str, default='CamVid/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--epochs', type=int, default=850, metavar='N', help='number of epochs to train (default: 850)')
parser.add_argument('--save_freq', type=int, default=100, metavar='N', help='save frequency (default: 10)')
parser.add_argument('--eval_freq', type=int, default=50, metavar='N', help='evaluation frequency (default: 5)')

parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--batch_size', type=int, default=3, metavar='N', help='input batch size (default: 3)')
parser.add_argument('--lr_init', type=float, default=1e-2, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr_decay', type=float, default=0.995, help='amount of learning rate decay per epoch (default: 0.995)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', type=str, choices=['RMSProp', 'SGD'], default='SGD')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')

parser.add_argument('--ft_start', type=int, default=750, help='begin fine-tuning with full sized images (default: 750)')
parser.add_argument('--ft_batch_size', type=int, default=1, help='fine-tuning batch size (default: 1)')
parser.add_argument('--ft_lr', type=float, default=1e-3, help='fine-tuning learning rate for RMSProp')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--use_weights', action='store_true', help='whether to use weighted loss')

parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--tau0', type=float, default=3., help='tau0')
parser.add_argument('--tau_min', type=float, default=1., help='tau_min')
parser.add_argument('--tau_anneal_rate', type=float, default=0.00002, help='tau_anneal_rate')
parser.add_argument('--test', action='store_true', default=False, help='If test')
parser.add_argument('--mcdropout', action='store_true', default=False, help='Use mc dropout for test')

args = parser.parse_args()

assert args.dataset == 'CamVid' # no other data loaders have been implemented

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Preparing directory %s' % args.dir)
create_exp_dir(args.dir, scripts_to_save=glob.glob('*.py') + glob.glob('utils/*.py') + glob.glob('models/*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.dir, 'log.txt'), mode="w+")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info(str(args))

with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

logging.info('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

if args.model == "DBSN":
    dbsn = True
else:
    dbsn = False

loaders, num_classes = camvid_loaders(args.data_path, args.batch_size, args.num_workers, ft_batch_size=args.ft_batch_size,
                    transform_train=model_cfg.transform_train, transform_test=model_cfg.transform_test,
                    joint_transform=model_cfg.joint_transform, ft_joint_transform=model_cfg.ft_joint_transform,
                    target_transform=model_cfg.target_transform)

logging.info('Beginning with cropped images')
train_loader = loaders['train']
num_batch = len(train_loader)
logging.info("num_batch: {}".format(num_batch))

logging.info('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()
model.apply(train_utils.weights_init)
logging.info('Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
ngpus = int(torch.cuda.device_count())
if ngpus > 1:
    model = nn.DataParallel(model)

if args.optimizer == 'RMSProp':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.lr_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, weight_decay = args.wd, momentum = 0.9)

if dbsn:
    args.nlayers = model_cfg.kwargs['nlayers_per_cell']
    alphas = Variable(1e-3*torch.randn(int((args.nlayers-1) * args.nlayers), 3).cuda(), requires_grad=True)
    betas = Variable(torch.ones(int((args.nlayers-1) * args.nlayers), 1).cuda()*math.log(1.), requires_grad=True)
    biOptimizer = train_utils.GumbelSoftmaxMOptimizer(model, args, num_batch, alphas, betas)
else:
    alphas, betas, biOptimizer = None, None, None

start_epoch = 1

criterion = train_utils.masked_ce_loss

if args.use_weights:
    class_weights = torch.FloatTensor([
    0.58872014284134, 0.51052379608154, 2.6966278553009,
    0.45021694898605, 1.1785038709641, 0.77028578519821, 2.4782588481903,
    2.5273461341858, 1.0122526884079, 3.2375309467316, 4.1312313079834]).cuda()

    criterion = partial(criterion, weight=class_weights)

if args.resume is not None:
    logging.info('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']+1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if dbsn:
        alphas = checkpoint['alphas'].clone()
        betas = checkpoint['betas'].clone()
        biOptimizer.tau = args.tau_min
    del checkpoint

    if args.test:
        if dbsn:
            print(torch.cat([F.softmax(alphas, 1), betas.exp()], 1).data.cpu().numpy())
        test_loss, test_err, test_iou = train_utils.test(model, loaders['test'], criterion, alphas, betas, biOptimizer,ngpus,dbsn,ntimes=100,mcdropout=args.mcdropout, dir=args.dir)
        logging.info('Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(test_loss, 1-test_err, test_iou))
        exit(0)

for epoch in range(start_epoch, args.epochs+1):
    if epoch % 50 is 1 and dbsn:
        print(torch.cat([F.softmax(alphas, 1), betas.exp()], 1).data.cpu().numpy())

    since = time.time()

    ### Train ###
    if epoch == args.ft_start+1:
        logging.info('Now replacing data loader with fine-tuned data loader.')
        train_loader = loaders['fine_tune']

    trn_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, biOptimizer, epoch, ngpus, dbsn)
    logging.info('Epoch {:d}    Train - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1-trn_err))
    # time_elapsed = time.time() - since
    # logging.info('Train Time {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))

    if epoch % args.eval_freq is 0:
        ### Test ###
        val_loss, val_err, val_iou = train_utils.test(model, loaders['val'], criterion, alphas, betas,biOptimizer, ngpus, dbsn)
        logging.info('Val - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(val_loss, 1-val_err, val_iou))

        test_loss, test_err, test_iou = train_utils.test(model, loaders['test'], criterion, alphas, betas, biOptimizer,ngpus,dbsn)
        logging.info('Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(test_loss, 1-test_err, test_iou))


    # time_elapsed = time.time() - since
    # logging.info('Total Time {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))

    ### Checkpoint ###
    if epoch % args.save_freq is 0 or epoch == args.epochs:
        logging.info('Saving model at Epoch: {}'.format(epoch))
        train_utils.save_checkpoint(dir=args.dir,
                            epoch=epoch,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            alphas=alphas,
                            betas=betas
                        )

    if args.optimizer=='RMSProp':
        ### Adjust Lr ###
        if epoch < args.ft_start:
            scheduler.step(epoch=epoch)
        else:
            #scheduler.step(epoch=-1) #reset to args.lr_init for fine-tuning
            train_utils.adjust_learning_rate(optimizer, args.ft_lr)

    elif args.optimizer=='SGD':
        lr = train_utils.schedule(epoch, args.lr_init, args.epochs)
        train_utils.adjust_learning_rate(optimizer, lr)

### Test set ###
test_loss, test_err, test_iou = train_utils.test(model, loaders['test'], criterion, alphas, betas, biOptimizer,ngpus,dbsn)
logging.info('Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(test_loss, 1-test_err, test_iou))
