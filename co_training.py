import numpy as np
import copy
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
from utils import AverageMeter
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    lab_dataset_1.data = lab_dataset_1.train_data[indices1]
    lab_dataset_1.targets = lab_dataset_1.train_labels[indices1]
    assert lab_dataset_1.train_labels.__len__() == lab_dataset_1.train_data.__len__() == k

    lab_dataset_2 = copy.deepcopy(train_dataset)
    lab_dataset_2.data= lab_dataset_2.train_data[indices2]
    lab_dataset_2.targets = lab_dataset_2.train_labels[indices2]
    assert lab_dataset_2.train_labels.__len__() == lab_dataset_2.train_data.__len__() == k

    # lab_dataset_1 = copy.deepcopy(train_dataset)[indices1]
    # lab_dataset_2 = copy.deepcopy(train_dataset)[indices2]
    other = [x.long() for x in other]
    unlab_dataset = copy.deepcopy(train_dataset)
    unlab_dataset.data = unlab_dataset.train_data[other]
    unlab_dataset.targets = unlab_dataset.train_labels[other]

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


def inference(model1, model2, val_loader):
    acc1_meter = AverageMeter()
    acc2_meter = AverageMeter()
    model1.eval()
    model2.eval()
    val_loader = tqdm(val_loader)
    for i, (img, gt) in enumerate(val_loader):
        img, gt = img.to(device), gt.to(device)
        pred1 = model1(img).max(1)[1]
        pred2 = model2(img).max(1)[1]
        acc1 = (pred1 == gt).sum().float() / gt.shape[0]
        acc2 = (pred2 == gt).sum().float() / gt.shape[0]
        # print(acc1)
        acc1_meter.update(acc1, gt.shape[0])
        acc2_meter.update(acc2, gt.shape[0])
        val_loader.set_postfix({"acc1": acc1_meter.avg, "acc2": acc2_meter.avg})

    model1.train()
    model2.train()
    return acc1_meter.avg, acc2_meter.avg


def train(model1, model2, seed, k=100, alpha=0.6, lr=0.002, beta2=0.99, num_epochs=150,
          batch_size=100, drop=0.5, std=0.15, fm1=16, fm2=32,
          divide_by_bs=False, w_norm=False, data_norm='pixelwise',
          early_stop=None, c=300, n_classes=10, max_epochs=80,
          max_val=30., ramp_up_mult=-5., n_samples=60000,
          print_res=True, device=None, epsilons=[0.01, 0.05], args=None, **kwargs):
    # retrieve data
    train_dataset, test_dataset = prepare_mnist()

    # build model
    model1.to(device)
    model2.to(device)

    # make data loaders
    batch_sizes = {  # 'lab': int(0.01 * batch_size),
        'lab': 60,
        # 'unlab': 1 - int(0.01 * batch_size),
        'unlab': 60,
        'test': 60}
    ## batch_size for lab: 1 unlabel: 0, 'test':100

    train_loaders, test_loader, indices = bundle_sample_train(train_dataset, test_dataset, batch_sizes,
                                                              k, n_classes, seed, shuffle_train=True)

    # setup param optimization
    optimizers = [torch.optim.Adam(model1.parameters(), lr=lr, betas=(0.9, 0.99)),
                  torch.optim.Adam(model2.parameters(), lr=lr, betas=(0.9, 0.99))]
    record = {}
    for epoch in range(num_epochs):
        t = timer()

        # evaluate unsupervised cost weight (TODO: check the w in the paper)
        ## this is the ramp function that weights on the unsupervised term.

        w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)
        # print(w)
        ## w turns out to increase from 0 to 0.05

        if (epoch + 1) % 10 == 0:
            print('unsupervised loss weight : {:.3f}'.format(w))

        # turn it into a usable pytorch object
        w = torch.FloatTensor([w]).to(device)

        acc1, acc2 = inference(model1, model2, test_loader)
        record[epoch] = {'acc1': acc1, 'acc2': acc2}
        pd.DataFrame(record).T.to_csv(str(args)+'.csv')

        train_loader= tqdm(train_loaders['lab_dataloader1'])
        for i,_ in enumerate(train_loader):
            ## pick up images.
            unlab_imgs, false_labels = image_batch_generator(train_loaders['unlab'], device=device)
            lab_img1, true_labels1 = image_batch_generator(train_loaders['lab_dataloader1'], device=device)
            lab_img2, true_labels2 = image_batch_generator(train_loaders['lab_dataloader2'], device=device)

            # add gradient on images
            lab_img1.requires_grad = True
            lab_img2.requires_grad = True
            if lab_img2.grad is not None:
                lab_img1.grad.zero_()
                lab_img2.grad.zero_()
            unlab_imgs.requires_grad = True
            if unlab_imgs.grad is not None:
                unlab_imgs.grad.zero_()

            optimizers[0].zero_grad()
            optimizers[1].zero_grad()

            lab_pred_1 = model1(lab_img1)
            lab_pred_2 = model2(lab_img2)

            loss1 = nn.CrossEntropyLoss()(lab_pred_1, true_labels1)
            loss2 = nn.CrossEntropyLoss()(lab_pred_2, true_labels2)
            if args.sup:
                supervisedLoss = loss1 + loss2
                supervisedLoss.backward()
                optimizers[0].step()
                optimizers[1].step()

            pred_unlab_1 = F.softmax(model1(unlab_imgs), 1)
            pred_unlab_2 = F.softmax(model2(unlab_imgs), 1)

            average_pred = (pred_unlab_1 + pred_unlab_2) / 2

            loss3 = F.kl_div(F.softmax(pred_unlab_1, 1).log(), average_pred.detach())
            loss4 = F.kl_div(F.softmax(pred_unlab_2, 1).log(), average_pred.detach())

            if args.jsd:

                jsdiv = loss3 + loss4
                unsupervisedLoss = w * jsdiv

                optimizers[0].zero_grad()
                optimizers[1].zero_grad()

                unsupervisedLoss.backward()
                optimizers[0].step()
                optimizers[1].step()

            ## to generate adversarial example for unlab

            if args.adv:

                if random() > 0.5:
                    adv_imgs_1 = fgsm_attack(lab_img1, epsilon=0.5, data_grad=lab_img1.grad.data).detach()
                else:
                    if unlab_imgs.grad is not None:
                        unlab_imgs.grad.zero_()
                    unlab_pred = model1(unlab_imgs)
                    unlab_mask = unlab_pred.max(1)[1]
                    loss = nn.CrossEntropyLoss()(unlab_pred, unlab_mask.detach())
                    loss.backward()
                    adv_imgs_1 = fgsm_attack(unlab_imgs, epsilon=0.5, data_grad=unlab_imgs.grad.data).detach()

                if random() > 0.5:
                    adv_imgs_2 = fgsm_attack(lab_img2, epsilon=0.5, data_grad=lab_img2.grad.data).detach()
                else:
                    if unlab_imgs.grad is not None:
                        unlab_imgs.grad.zero_()
                    unlab_pred = model2(unlab_imgs)
                    unlab_mask = unlab_pred.max(1)[1]
                    loss = nn.CrossEntropyLoss()(unlab_pred, unlab_mask.detach())
                    loss.backward()
                    adv_imgs_2 = fgsm_attack(unlab_imgs, epsilon=0.5, data_grad=unlab_imgs.grad.data).detach()

                pred_11 = F.softmax(model1(lab_img1), 1)
                pred_22 = F.softmax(model2(lab_img2), 1)
                pred_21 = F.softmax(model2(adv_imgs_1), 1)
                pred_12 = F.softmax(model1(adv_imgs_2), 1)
                loss5 = F.kl_div(pred_11.log(), pred_21.detach())
                loss6 = F.kl_div(pred_22.log(), pred_12.detach())

                adv_loss = w / 10 * (loss5 + loss6)

                optimizers[0].zero_grad()
                optimizers[1].zero_grad()

                adv_loss.backward()

                optimizers[0].step()
                optimizers[1].step()


    return record
