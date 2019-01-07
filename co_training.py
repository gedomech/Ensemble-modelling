import numpy as np
import copy
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import calc_metrics, prepare_mnist, weight_schedule, fgsm_attack, image_batch_generator


def bundle_sample_train(train_dataset, test_dataset, batch_sizes, k, n_classes,
                        seed, shuffle_train=True, return_idxs=True):
    n = len(train_dataset)
    rrng = np.random.RandomState(seed)

    cpt = 0
    indices1 = torch.zeros(k)
    indices2 = torch.zeros(k)
    other = torch.zeros(n - 2 * k)
    card = k // n_classes

    for i in range(n_classes):
        class_items = (train_dataset.train_labels == i).nonzero()[:, 0]
        n_class = len(class_items)
        rd = np.random.permutation(np.arange(n_class))
        indices1[i * card: (i + 1) * card] = class_items[rd[:card]]
        indices2[i * card: (i + 1) * card] = class_items[rd[card:2 * card]]
        other[cpt: cpt + n_class - 2 * card] = class_items[rd[2 * card:]]
        cpt += n_class - 2 * card

    lab_dataset_1 = copy.deepcopy(train_dataset)[indices1]
    lab_dataset_2 = copy.deepcopy(train_dataset)[indices2]

    other = other.long()
    train_dataset.train_labels[other] = -1
    unlab_dataset = train_dataset[other]

    train_lab_loader1 = torch.utils.data.DataLoader(dataset=lab_dataset_1,
                                                    batch_size=batch_sizes['lab'],
                                                    num_workers=4,
                                                    shuffle=shuffle_train)
    train_lab_loader2 = torch.utils.data.DataLoader(dataset=lab_dataset_2,
                                                    batch_size=batch_sizes['lab'],
                                                    num_workers=4,
                                                    shuffle=shuffle_train)
    train_unlab_loader = torch.utils.data.DataLoader(dataset=unlab_dataset,
                                                     batch_size=batch_sizes['unlab'],
                                                     num_workers=4,
                                                     shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_sizes['test'],
                                              num_workers=4,
                                              shuffle=False)
    train_loaders = {'lab_dataloader1': train_lab_loader1,
                     'lab_dataloader2': train_lab_loader2,
                     'unlab': train_unlab_loader}
    if return_idxs:
        return train_loaders, test_loader, indices1, indices2
    return train_loaders, test_loader


def cotraining_loss(out_lst, adv_lst, labels, device, lambda_cot=0.3, lambda_diff=0.3):
    """
    Loss function implementation following paper by Qiao et al. (https://arxiv.org/abs/1803.05984)
    TODO: check in the paper the configuration of lambda hyperparameters
    :param out_lst: iterable object containing the outputs of each model given input sample (x)
    :param adv_lst: iterable object containing the outputs of each model for the adversarial examples given x
    :param labels: list of the corresponding labels from each bundle data stream (labels=-1 are unlabeled samples)
    :param device:
    :param lambda_cot:
    :param lambda_diff:
    :return:
    """
    def sup_loss(out_lst, labels):
        """

        :param out_lst:
        :param labels:
        :return: supervised loss value
        """
        loss = 0
        cond = (labels >= 0)
        nnz = torch.nonzero(cond)
        nbsup = len(nnz)
        # check if labeled samples in batch, return 0 if none
        if nbsup > 0:
            for out in out_lst:
                masked_outputs = torch.index_select(out, dim=0, index=nnz.view(nbsup))
                masked_labels = labels[cond]
                loss += F.cross_entropy(masked_outputs, masked_labels)
            return loss / len(out_lst), nbsup
        return torch.FloatTensor([0.]).to(device), loss

    def cot_loss(out_lst):
        """
        co-training loss based on Jensen-Shannon Divergence (JSD), computed between two distributions as:
                    JSD(X||Y) = 0.5 * (DKL(X||M) + DKL(Y||M))
        where DKL is the Kullback-Leibler Divergence and M = 0.5 * (X + Y) is the mixture distribution
        :param out_lst: iterable object containing the outputs of each model
        :return:
        """
        preds = torch.cat(out_lst)
        prob_distribs = F.softmax(preds, dim=1)
        N, BS, C = prob_distribs.shape
        mixture_distrib = prob_distribs.mean(0, keepdim=True).expand(N, BS, C)
        return F.kl_div(F.log_softmax(preds, dim=1), mixture_distrib, reduction='mean')

    def diff_loss(out_lst, adv_lst):
        """
        View Difference Constrain (VDC) loss based on classifiers prediction given the input sample (x) and the
        adversarial example g(x):
                    L_diff(x) = H(p1(x), p2(g1(x))) + H(p2(x), p1(g2(x)))
        where H is the Cross Entropy
        :param out_lst: iterable object containing the outputs of each model given input sample (x)
        :param adv_lst: iterable object containing the outputs of each model for the adversarial examples given x
        :return: difference loss value
        """
        adv_lst = reversed(adv_lst)
        loss = 0
        for idx, out in enumerate(out_lst):
            loss += F.cross_entropy(out, adv_lst[idx])
        return loss / len(out_lst)

    # compute supervised term
    sup_loss, nbsup = sup_loss(out_lst, labels)

    # get output from generated adversarial examples and compute VDC term
    unsup_diff_loss = diff_loss(out_lst, adv_lst)

    # retrieve only the unlabeled samples
    unlab_cond = (labels < 0)
    nnz = torch.nonzero(unlab_cond)
    nbunsup = len(nnz)
    for idx in enumerate(out_lst):
        out_lst[idx] = torch.index_select(out_lst[idx], dim=0, index=nnz.view(nbsup))

    unsup_cot_loss = 0
    if nbunsup > 0:
        # compute co-training term
        unsup_cot_loss = cot_loss(out_lst)

    total_loss = sup_loss + lambda_cot * unsup_cot_loss + lambda_diff * unsup_diff_loss

    return total_loss, sup_loss, unsup_cot_loss, unsup_diff_loss, nbsup


def train(model1, model2, seed, k=100, alpha=0.6, lr=0.002, beta2=0.99, num_epochs=150,
          batch_size=100, drop=0.5, std=0.15, fm1=16, fm2=32,
          divide_by_bs=False, w_norm=False, data_norm='pixelwise',
          early_stop=None, c=300, n_classes=10, max_epochs=80,
          max_val=30., ramp_up_mult=-5., n_samples=60000,
          print_res=True, device=None, epsilons = [0.01, 0.05], **kwargs):
    # retrieve data
    train_dataset, test_dataset = prepare_mnist()
    ntrain = len(train_dataset)

    # build model
    model1.to(device)
    model2.to(device)

    # make data loaders
    batch_sizes = {'lab': int(0.01 * batch_size),
                   'unlab': 1 - int(0.01 * batch_size),
                   'test': batch_size}
    train_loaders, test_loader, indices = bundle_sample_train(train_dataset, test_dataset, batch_sizes,
                                                              k, n_classes, seed, shuffle_train=False)

    # setup param optimization
    optimizers = [torch.optim.Adam(model1.parameters(), lr=lr, betas=(0.9, 0.99)),
                  torch.optim.Adam(model2.parameters(), lr=lr, betas=(0.9, 0.99))]

    # train
    models = [model1.train(), model2.train()]
    losses, sup_losses, unsup_losses = [], [], []
    best_loss = 20.

    for epoch in range(num_epochs):
        t = timer()

        # evaluate unsupervised cost weight (TODO: check the w in the paper)
        w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)

        if (epoch + 1) % 10 == 0:
            print('unsupervised loss weight : {}'.format(w))

        # turn it into a usable pytorch object
        w = torch.FloatTensor([w]).to(device)

        l, supl, unsupl = [], [], []
        for i in range(len(train_loaders['unlab'])):

            unlab_imgs, false_labels = image_batch_generator(train_loaders['unlab'], device=device)
            lab_img1, true_labels1 = image_batch_generator(train_loaders['lab_dataloader1'], device=device)
            lab_img2, true_labels2 = image_batch_generator(train_loaders['lab_dataloader2'], device=device)

            # creating the bundle data streams for each model
            imgs_1, labels_1 = torch.cat(lab_img1, unlab_imgs), torch.cat(true_labels1, false_labels)
            imgs_2, labels_2 = torch.cat(lab_img2, unlab_imgs), torch.cat(true_labels2, false_labels)
            bundles = [(imgs_1, labels_1), (imgs_2, labels_2)]

            # collect datagrads
            data_grads = [imgs_1.grad.data, imgs_2.grad.data]

            outs, adv_outs = [], []
            for idx, model in enumerate(models):
                optimizers[idx].zero_grad()
                # get output from bundle data streams
                outs.append(model(bundles[idx][0]))
                # generate adversarial examples
                perturbed_data = fgsm_attack(F.log_softmax(bundles[idx][0], dim=1),
                                             epsilons[idx], data_grads[idx])
                # re-classify the perturbed image
                adv_outs.append(model(perturbed_data))

            # calculate losses
            loss, suploss, cotloss, diffloss, nbsup = cotraining_loss(outs, adv_outs, [labels_1, labels_2], device,
                                                                      lambda_cot=0., lambda_diff=0.)
