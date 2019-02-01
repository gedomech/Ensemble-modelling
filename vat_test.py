"""

"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

from torch import Tensor
from typing import List, Tuple
from munkres import Munkres

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import itertools

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Data
print('==> Preparing data..')


class MyDataset(torch.utils.data.Dataset):
    # new dataset class that allows to get the sample indices of mini-batch
    def __init__(self, root, download, train, transform):
        self.MNIST = torchvision.datasets.MNIST(root=root,
                                                download=download,
                                                train=train,
                                                transform=transform)

    def __getitem__(self, index):
        data, target = self.MNIST[index]
        return data, target, index

    def __len__(self):
        return len(self.MNIST)


def myloader(args, dataset="mnist"):
    # TODO: implement for another datasets
    if dataset == "mnist":
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = MyDataset(root='./data', train=True, download=True, transform=transform_train)
        testset = MyDataset(root='./data', train=False, download=False, transform=transform_train)
        trainset = trainset + testset
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        tot_cl = 10
        input_size = 28 ** 2
        data_length = len(trainset)
        return trainloader, testloader, tot_cl, input_size, data_length


# Deep Neural Network
class Net(nn.Module):
    def __init__(self, input_size, tot_cl):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 1200)
        torch.nn.init.normal_(self.fc1.weight, std=0.1 * math.sqrt(2 / (input_size)))
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(1200, 1200)
        torch.nn.init.normal_(self.fc2.weight, std=0.1 * math.sqrt(2 / 1200))
        self.fc2.bias.data.fill_(0)
        self.fc3 = nn.Linear(1200, tot_cl)
        torch.nn.init.normal_(self.fc3.weight, std=0.0001 * math.sqrt(2 / 1200))
        self.fc3.bias.data.fill_(0)
        self.bn1 = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn1_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)
        self.bn2 = nn.BatchNorm1d(1200, eps=2e-5)
        self.bn2_F = nn.BatchNorm1d(1200, eps=2e-5, affine=False)

    def forward(self, x, update_batch_stats=True):
        if not update_batch_stats:
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.bn1_F(x) * self.bn1.weight + self.bn1.bias
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2_F(x) * self.bn2.weight + self.bn2.bias
            x = F.relu(x)
            x = self.fc3(x)
            return x
        else:
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x


class VATGenerator(object):
    def __init__(self, net: nn.Module, xi=1e-2, eplision=1.0, ip=1, axises: List[int] = [1, 2, 3]) -> None:
        """VAT generator based on https://arxiv.org/abs/1704.03976
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATGenerator, self).__init__()
        self.xi = xi
        self.eps = eplision
        self.ip = ip
        self.net = net
        self.axises = axises

    #         self.entropy = Entropy_2D()  # type: Tensor # shape:

    @staticmethod
    def _l2_normalize(d) -> Tensor:
        # d = d.cpu().numpy()
        # d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        # return torch.from_numpy(d)
        # d_reshaped = d.view(d.shape[0], 2)
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-16
        # print('d_noise shape:', d.shape)

        #         print(d.view(d.shape[0], -1).norm(dim=1))
        #         assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), torch.ones(d.shape[0]).to(d.device))

        return d

    @staticmethod
    def kl_div_with_logit(q_logit, p_logit, axises):
        '''
        :param q_logit:it is like the y in the ce loss
        :param p_logit: it is the logit to be proched to q_logit
        :return:
        '''
        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq)[:, axises].sum(dim=1)
        qlogp = (q * logp)[:, axises].sum(dim=1)
        # assert (qlogq - qlogp<0).sum()==0

        return qlogq - qlogp

    @staticmethod
    def l2_loss2d(q_logit, p_logit, axises):
        q_prob = F.softmax(q_logit, 1)
        p_prob = F.softmax(p_logit, 1)
        loss = (q_prob - p_prob).pow(2)[:, axises].sum(dim=1)
        return loss

    @staticmethod
    def l1_loss2d(q_logit, p_logit, axises):
        q_prob = F.softmax(q_logit, 1)
        p_prob = F.softmax(p_logit, 1)
        loss = torch.abs((q_prob - p_prob))[:, axises].sum(dim=1)
        return loss

    def __call__(self, img: Tensor, loss_name='kl') -> Tuple[Tensor, Tensor]:
        tra_state = self.net.training
        self.net.eval()
        with torch.no_grad():
            pred = self.net(img)

        # prepare random unit tensor
        d = torch.Tensor(img.size()).normal_()  # 所有元素的std =1, average = 0
        d = self._l2_normalize(d).to(img.device)
        assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1),
                              torch.ones(d.shape[0]).to(img.device)), 'The L2 normalization fails'
        self.net.zero_grad()
        for _ in range(self.ip):
            d = self.xi * self._l2_normalize(d).to(img.device)
            d.requires_grad = True
            y_hat = self.net(img + d)
            delta_kl: torch.Tensor
            # Here the pred is the reference as y in cross-entropy.

            if loss_name == 'kl':
                delta_kl = self.kl_div_with_logit(pred.detach(), y_hat, self.axises)  # B/H/W
            elif loss_name == 'l2':
                delta_kl = self.l2_loss2d(pred.detach(), y_hat, self.axises)  # B/H/W
            elif loss_name == 'l1':
                delta_kl = self.l1_loss2d(pred.detach(), y_hat, self.axises)  # B/H/W
            else:
                raise NotImplementedError

            # todo: the mask

            delta_kl.mean().backward()

            d = d.grad.data.clone().cpu()
            self.net.zero_grad()
        ##
        d = self._l2_normalize(d).to(img.device)
        r_adv = self.eps * d
        # compute lds
        img_adv = img + r_adv.detach()
        if tra_state:
            self.net.train()
        assert self.net.training == tra_state
        # img_adv = torch.clamp(img_adv, 0, 1)

        return img_adv.detach(), r_adv.detach()


def compute_accuracy(y_pred, y_t, tot_cl):
    # compute the accuracy using Hungarian algorithm
    m = Munkres()
    mat = np.zeros((tot_cl, tot_cl))
    for i in range(tot_cl):
        for j in range(tot_cl):
            mat[i][j] = np.sum(np.logical_and(y_pred == i, y_t == j))
    indexes = m.compute(-mat)

    corresp = []
    for i in range(tot_cl):
        corresp.append(indexes[i][1])

    pred_corresp = [corresp[int(predicted)] for predicted in y_pred]
    acc = np.sum(pred_corresp == y_t) / float(len(y_t))
    return acc


def train(args, net, device, use_cuda, trainloader, optimizer):
    criterion = nn.CrossEntropyLoss()

    net.train()
    running_loss = 0.0
    MI_loss = 0.0
    SC_loss = 0.0
    VAT_loss = 0.0
    loss_vat = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels, ind = data
        if use_cuda:
            inputs, labels, nearest_dist, ind = inputs.to(device), labels.to(device), nearest_dist.to(device), ind.to(
                device)

        # computation of supervised loss term
        pred = net(inputs)
        loss_sup = criterion(pred, labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        if args.vat == "VATLoss":
            inputs.requires_grad = True
            real_pred = net(inputs)
            try:
                inputs.grad.zeros()
            except:
                pass
            adv_inputs, noise = VATGenerator(net, eplision=0.1, axises=[0, 9])(inputs, loss_name='kl')
            adv_pred = net(adv_inputs)
            loss_vat = F.kl_div(F.log_softmax(adv_pred, 1), F.softmax(real_pred, 1).detach())

        loss = loss_sup + 0.01 * loss_vat

        # backward + optimize
        loss.backward()
        optimizer.step()

        # loss accumulation
        running_loss += loss.item()
        VAT_loss += loss_vat.item()

    return running_loss / (i + 1),  MI_loss / (i + 1), SC_loss / (i + 1), VAT_loss / (i + 1)


def test(net, use_cuda, device, testloader, data_length, tot_cl, args):
    net.eval()
    p_pred = np.zeros((data_length, tot_cl))
    y_pred = np.zeros(data_length)
    y_t = np.zeros(data_length)
    acc = 0.
    MI = 0.
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels, ind = data
            # inputs = inputs.view(inputs.shape[0],-1)
            if use_cuda:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = F.softmax(net(inputs), dim=1)

            y_pred[i * args.batch_size:(i + 1) * args.batch_size] = torch.argmax(outputs, dim=1).cpu().numpy()
            p_pred[i * args.batch_size:(i + 1) * args.batch_size, :] = outputs.detach().cpu().numpy()
            y_t[i * args.batch_size:(i + 1) * args.batch_size] = labels.cpu().numpy()
        acc = compute_accuracy(y_pred, y_t, tot_cl)
        MI = normalized_mutual_info_score(y_t, y_pred)
    return acc, MI


def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--batch_size', '-b', default=256, type=int, help='size of the batch during training')
    parser.add_argument('--lam', type=float,
                        help='trade-off parameter for mutual information and smooth regularization', default=0.1)
    parser.add_argument('--mu', type=float,
                        help='trade-off parameter for entropy minimization and entropy maximization', default=4)
    parser.add_argument('--prop_eps', type=float, help='epsilon', default=0.25)
    parser.add_argument('--hidden_list', type=str, help='hidden size list', default='1200-1200')
    parser.add_argument('--n_epoch', type=int, help='number of epoches when maximizing', default=50)
    parser.add_argument('--n_trials', type=int, help='number of trials', default=12)
    parser.add_argument('--dataset', type=str, help='which dataset to use', default='mnist')
    parser.add_argument('--vat', type=str, help='which vat function to use: mine or the one from VATLoss',
                        default='VATLoss')
    parser.add_argument('--lam_sc', type=float, help='trade-off parameter for SC and smooth regularization',
                        default=0.1)
    parser.add_argument('--k_near', type=float,
                        help='Kth nearest neighbor considered when computing the affinity matrix', default=25)
    args = parser.parse_args()
    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Training
    print('==> loading data..')
    trainloader, testloader, tot_cl, input_size, data_length = myloader(args, args.dataset)
    print('==> Start training..')

    acc_v = np.zeros((1, args.n_trials))
    best_acc_v = np.zeros((1, args.n_trials))
    mi_v = np.zeros((1, args.n_trials))
    corres_mi_v = np.zeros((1, args.n_trials))

    for trial in range(args.n_trials):
        # TODO: implement for another datasets
        if args.dataset == "mnist":
            net = Net(input_size, tot_cl)
            if use_cuda:
                net.to(device)
        else:
            print("The dataset is not supported.")
            raise NotImplementedError

        best_acc = 0.
        corres_mi = 0.
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        for epoch in range(args.n_epoch):
            running_loss, MI_loss, SC_loss, VAT_loss = train(args, net, device, use_cuda, trainloader, optimizer)

            acc, mi = test(net, use_cuda, device, testloader, data_length, tot_cl, args)
            if acc > best_acc:
                best_acc = acc
                corres_mi = mi
            if 1:
                print("trial: ", trial + 1, "epoch: ", epoch + 1, "\t total lost = {:.4f} ".format(running_loss),
                      "\t MI lost = {:.4f} ".format(MI_loss), "\t SC lost = {:.4f} ".format(SC_loss),
                      "\t VAT lost = {:.4f} ".format(VAT_loss), "\t MI = {:.4f}".format(mi),
                      "\t acc = {:.4f} ".format(acc))
            acc_v[0, trial] = acc
            mi_v[0, trial] = mi
            best_acc_v[0, trial] = best_acc
            corres_mi_v[0, trial] = corres_mi

    print('==> print results..')
    mean_acc = 0.
    mean_mi = 0.
    std_acc = 0.
    std_mi = 0.
    mean_best_acc = 0.
    mean_corres_mi = 0.
    mean_acc = np.mean(acc_v)
    mean_mi = np.mean(mi_v)
    mean_best_acc = np.mean(best_acc_v)
    mean_corres_mi = np.mean(corres_mi_v)
    std_acc = np.std(acc_v)
    std_mi = np.std(mi_v)
    std_best_acc = np.std(best_acc_v)

    print("acc_v", acc_v)
    print("best_acc_v", best_acc_v)
    print("n trial: ", args.n_trials, "n epoch: ", args.n_epoch, "\t mean acc = {:.4f} ".format(mean_acc),
          "\t std acc = {:.4f} ".format(std_acc), "\t mean best acc = {:.4f} ".format(mean_best_acc),
          "\t std best acc = {:.4f} ".format(std_best_acc), "\t mean MI = {:.4f} ".format(mean_mi),
          "\t std MI = {:.4f} ".format(std_mi))
    print('==> Finished Training..')


if __name__ == '__main__':
    main()
