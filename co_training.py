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
        indices2[i * card: (i + 1) * card] = class_items[rd[card:2 * card]].long()
        other[cpt: cpt + n_class - 2 * card] = class_items[rd[2 * card:]].long()
        cpt += n_class - 2 * card
    indices1 = [x.long() for x in indices1]
    indices2 = [x.long() for x in indices2]
    assert len(set(indices1) & set(indices2)) == 0
    assert len(set(indices1) & set(other)) == 0
    assert len(set(indices2) & set(other)) == 0
    assert len(set(indices1) | set(indices2) | set(other)) == 60000

    lab_dataset_1 = copy.deepcopy(train_dataset)
    lab_dataset_1.train_data = lab_dataset_1.train_data[indices1]
    lab_dataset_1.train_labels = lab_dataset_1.train_labels[indices1]
    assert lab_dataset_1.train_labels.__len__() == lab_dataset_1.train_data.__len__() == k

    lab_dataset_2 = copy.deepcopy(train_dataset)
    lab_dataset_2.train_data = lab_dataset_2.train_data[indices2]
    lab_dataset_2.train_labels = lab_dataset_2.train_labels[indices2]
    assert lab_dataset_2.train_labels.__len__() == lab_dataset_2.train_data.__len__() == k

    # lab_dataset_1 = copy.deepcopy(train_dataset)[indices1]
    # lab_dataset_2 = copy.deepcopy(train_dataset)[indices2]
    other = [x.long() for x in other]
    unlab_dataset = copy.deepcopy(train_dataset)
    unlab_dataset.train_data = unlab_dataset.train_data[other]
    unlab_dataset.train_labels = unlab_dataset.train_labels[other]

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
        return train_loaders, test_loader, (indices1, indices2)
    return train_loaders, test_loader, (None, None)


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
          print_res=True, device=None, epsilons=[0.01, 0.05], **kwargs):
    # retrieve data
    train_dataset, test_dataset = prepare_mnist()
    ntrain = len(train_dataset)

    # build model
    model1.to(device)
    model2.to(device)

    # make data loaders
    batch_sizes = {  # 'lab': int(0.01 * batch_size),
        'lab': 10,
        # 'unlab': 1 - int(0.01 * batch_size),
        'unlab': 10,
        'test': 10}
    ## batch_size for lab: 1 unlabel: 0, 'test':100

    train_loaders, test_loader, indices = bundle_sample_train(train_dataset, test_dataset, batch_sizes,
                                                              k, n_classes, seed, shuffle_train=True)

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
        ## this is the ramp function that weights on the unsupervised term.

        w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)
        ## w turns out to increase from 0 to 0.05

        if (epoch + 1) % 10 == 0:
            print('unsupervised loss weight : {}'.format(w))

        # turn it into a usable pytorch object
        w = torch.FloatTensor([w]).to(device)

        l, supl, unsupl = [], [], []
        for i in range(len(train_loaders['unlab'])):

            ## pick up images.
            unlab_imgs, false_labels = image_batch_generator(train_loaders['unlab'], device=device)
            lab_img1, true_labels1 = image_batch_generator(train_loaders['lab_dataloader1'], device=device)
            lab_img2, true_labels2 = image_batch_generator(train_loaders['lab_dataloader2'], device=device)

            # creating the bundle data streams for each model
            imgs_1, labels_1 = torch.cat((lab_img1, unlab_imgs), dim=0), torch.cat((true_labels1, false_labels), dim=0)
            imgs_2, labels_2 = torch.cat((lab_img2, unlab_imgs), dim=0), torch.cat((true_labels2, false_labels), dim=0)
            bundles = [(imgs_1, labels_1), (imgs_2, labels_2)]

            ## add gradient on images
            lab_img1.requires_grad = True
            lab_img2.requires_grad = True
            if lab_img2.grad is not None:
                lab_img1.grad.zero_()
                lab_img2.grad.zero_()


            # collect datagrads
            # data_grads = [imgs_1.grad.data, imgs_2.grad.data]

            optimizers[0].zero_grad()
            optimizers[1].zero_grad()

            lab_pred_1 = model1(lab_img1)
            lab_pred_2 = model2(lab_img2)

            loss1 = nn.CrossEntropyLoss()(lab_pred_1, true_labels1)
            loss2 = nn.CrossEntropyLoss()(lab_pred_2, true_labels2)

            loss1.backward()
            optimizers[0].step()
            loss2.backward()
            optimizers[1].step()

            ## adversarial loss
            adv_imgs_1 = fgsm_attack(lab_img1, epsilon=0.5, data_grad=lab_img1.grad.data).detach()
            adv_imgs_2 = fgsm_attack(lab_img2, epsilon=0.5, data_grad=lab_img2.grad.data).detach()

            optimizers[0].zero_grad()
            optimizers[1].zero_grad()
            pred_11 = model1(lab_img1)
            pred_22 = model2(lab_img2)
            pred_21 = model2(adv_imgs_1)
            pred_12 = model1(adv_imgs_2)
            loss3 = F.kl_div(pred_11, pred_21.detach())
            loss4 = F.kl_div(pred_22, pred_12.detach())

            loss3.backward()
            optimizers[0].step()
            loss4.backward()
            optimizers[1].step()

            ## different loss
            optimizers[0].zero_grad()
            optimizers[1].zero_grad()

            pred_unlab_1 = model1(unlab_imgs)
            pred_unlab_2 = model2(unlab_imgs)

            loss5 = F.kl_div(pred_unlab_1,pred_unlab_2.detach())
            loss6 = F.kl_div(pred_unlab_2,pred_unlab_1.detach())

            loss = loss5+loss6
            loss.backward(retain_graph=True)
            optimizers[0].zero_grad()
            optimizers[1].zero_grad()




            # data_grads = [imgs_1.grad.data, imgs_2.grad.data]
            #
            # outs, adv_outs = [], []
            # for idx, model in enumerate(models):
            #     optimizers[idx].zero_grad()
            #     # get output from bundle data streams
            #     outs.append(model(bundles[idx][0]))
            #     # generate adversarial examples
            #     perturbed_data = fgsm_attack(F.log_softmax(bundles[idx][0], dim=1),
            #                                  epsilons[idx], data_grads[idx])
            #     # re-classify the perturbed image
            #     adv_outs.append(model(perturbed_data))
            #
            # # calculate losses
            # loss, suploss, cotloss, diffloss, nbsup = cotraining_loss(outs, adv_outs, [labels_1, labels_2], device,
            #                                                           lambda_cot=0., lambda_diff=0.)
