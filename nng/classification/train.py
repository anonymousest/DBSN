from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.base_train import BaseTrain
from tqdm import tqdm

import numpy as np
from classification.misc.data_loader import Transpose
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
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

class Trainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()
            self.test_epoch()

            if (cur_epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.model.save(self.sess)

    def train_epoch(self):
        loss_list = []
        acc_list = []
        for itr, (x, y) in enumerate(tqdm(self.train_loader)):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.n_particles: self.config.train_particles
            }
            feed_dict.update({self.model.is_training: True})
            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            feed_dict.update({self.model.is_training: False})  # Note: that's important
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

            cur_iter = self.model.global_step_tensor.eval(self.sess)

            if cur_iter % self.config.get('TCov', 10) == 0 and \
                    self.model.cov_update_op is not None:
                self.sess.run([self.model.cov_update_op], feed_dict=feed_dict)

                if self.config.optimizer == "diag":
                    self.sess.run([self.model.var_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.get('TInv', 200) == 0 and \
                    self.model.inv_update_op is not None:
                self.sess.run([self.model.inv_update_op, self.model.var_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.get('TEigen', 200) == 0 and \
                    self.model.eigen_basis_update_op is not None:
                self.sess.run([self.model.eigen_basis_update_op, self.model.var_update_op], feed_dict=feed_dict)

                if self.config.kfac_init_after_basis:
                    self.sess.run(self.model.re_init_kfac_scale_op)

            if cur_iter % self.config.get('TScale', 10) == 0 and \
                    self.model.scale_update_op is not None:
                self.sess.run([self.model.scale_update_op, self.model.var_scale_update_op], feed_dict=feed_dict)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("train | loss: %5.6f | accuracy: %5.6f" % (float(avg_loss), float(avg_acc)))

        # Summarize
        summaries_dict = dict()
        summaries_dict['train_loss'] = avg_loss
        summaries_dict['train_acc'] = avg_acc

        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def test_epoch(self):
        loss_list = []
        acc_list = []
        for (x, y) in self.test_loader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: self.config.test_particles
            }
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("test | loss: %5.6f | accuracy: %5.6f\n" % (float(avg_loss), float(avg_acc)))

        # Summarize
        summaries_dict = dict()
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc

        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def test_final(self):
        self.model.load(self.sess)

        loss_list = []
        acc_list = []

        labels_list = []
        logits_list = []
        for (x, y) in self.test_loader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: 30
            }
            loss, acc, logits_batch = self.sess.run([self.model.loss, self.model.acc, self.model.logits], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

            labels_list.append(y)
            logits_list.append(torch.from_numpy(logits_batch))

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        ece = _ECELoss()(torch.cat(logits_list, 0), torch.cat(labels_list, 0), 'experiments/' + self.config.dataset + '/' + self.config.exp_name)
        print("test | loss: %5.6f | accuracy: %5.6f | ece: %5.6f\n" % (float(avg_loss), float(avg_acc), ece))

    def test_svhn_ent(self):
        self.model.load(self.sess)

        kwargs = {'num_workers': 4}
        svhn_testloader = torch.utils.data.DataLoader(
            dset.SVHN(root='../svhn', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if self.config.dataset == "cifar10" else transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                Transpose()
            ])),
            batch_size=self.config.test_batch_size, shuffle=False, **kwargs)

        loss_list = []
        acc_list = []
        ent_list = []
        for (x, y) in svhn_testloader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: 30
            }
            loss, acc, ents = self.sess.run([self.model.loss, self.model.acc, self.model.ent], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)
            ent_list.append(ents)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        ents = np.concatenate(ent_list, 0)
        np.save('experiments/' + self.config.dataset + '/' + self.config.exp_name + "/svhn_entropies.npy", ents)
        print("test | loss: %5.6f | accuracy: %5.6f\n" % (float(avg_loss), float(avg_acc)))

    def attack(self, bim=False):

        self.model.load(self.sess)

        if self.config.dataset == "cifar10":
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                Transpose()
            ])
            testset = dset.CIFAR10(root=self.config.data_path, train=False, download=True, transform=test_transform)
            testloader = torch.utils.data.DataLoader(testset,
                                                     batch_size=self.config.test_batch_size,
                                                     shuffle=False,
                                                     num_workers=self.config.num_workers)

        elif self.config.dataset == "cifar100":
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                Transpose()
            ])
            testset = dset.CIFAR100(root=self.config.data_path, train=False, download=True, transform=test_transform)
            testloader = torch.utils.data.DataLoader(testset,
                                                     batch_size=self.config.test_batch_size,
                                                     shuffle=False,
                                                     num_workers=self.config.num_workers)

        print(len(testloader))
        for magnitude_ in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
            advxs_list = []
            labels_list = []
            if bim:
                for (x, y) in testloader:

                    for ite in range(3):
                        feed_dict = {
                            self.model.inputs: x if ite == 0 else advx,
                            self.model.targets: y,
                            self.model.is_training: False,
                            self.model.n_particles: 30,
                            self.model.magnitude: magnitude_/3.
                        }
                        [advx] = self.sess.run([self.model.inputs_adv], feed_dict=feed_dict)
                    advxs_list.append(advx)
                    labels_list.append(y.numpy())
            else:
                for (x, y) in testloader:

                    feed_dict = {
                        self.model.inputs: x,
                        self.model.targets: y,
                        self.model.is_training: False,
                        self.model.n_particles: 30,
                        self.model.magnitude: magnitude_
                    }
                    [advx] = self.sess.run([self.model.inputs_adv], feed_dict=feed_dict)
                    advxs_list.append(advx)
                    labels_list.append(y.numpy())

            testset_adv = torch.utils.data.TensorDataset(torch.from_numpy(np.concatenate(advxs_list, 0)), torch.from_numpy(np.concatenate(labels_list, 0)))
            testloader_adv = torch.utils.data.DataLoader(testset_adv,
                                                     batch_size=self.config.test_batch_size,
                                                     shuffle=False,
                                                     num_workers=self.config.num_workers)
            #print(len(testloader_adv))
            loss_list = []
            acc_list = []
            ent_list = []
            for (x, y) in testloader_adv:
                feed_dict = {
                    self.model.inputs: x,
                    self.model.targets: y,
                    self.model.is_training: False,
                    self.model.n_particles: 30
                }
                loss, acc, ents = self.sess.run([self.model.loss, self.model.acc, self.model.ent], feed_dict=feed_dict)
                loss_list.append(loss)
                acc_list.append(acc)
                ent_list.append(ents)

            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)
            avg_ent = np.concatenate(ent_list, 0).mean()

            print('2019-09-01 10:26:52,395 Test set: epsilon: {} average loss: {:.4f}, entropy: {:.4f}, Error: {}/10000 ({:.2f}%)'.format(
                magnitude_, avg_loss, avg_ent, int(10000*(1-avg_acc)), (1-avg_acc)*100))
