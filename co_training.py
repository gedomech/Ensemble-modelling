import numpy as np
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import calc_metrics, prepare_mnist, weight_schedule, fgsm_attack


def sample_train(train_dataset, test_dataset, batch_size, k, n_classes,
                 seed, shuffle_train=True, return_idxs=True):
    n = len(train_dataset)
    rrng = np.random.RandomState(seed)

    cpt = 0
    indices = torch.zeros(k)
    other = torch.zeros(n - k)
    card = k // n_classes

    for i in range(n_classes):
        class_items = (train_dataset.train_labels == i).nonzero()[:, 0]
        n_class = len(class_items)
        rd = np.random.permutation(np.arange(n_class))
        indices[i * card: (i + 1) * card] = class_items[rd[:card]]
        other[cpt: cpt + n_class - card] = class_items[rd[card:]]
        cpt += n_class - card

    other = other.long()
    train_dataset.train_labels[other] = -1

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=False)

    if return_idxs:
        return train_loader, test_loader, indices
    return train_loader, test_loader


def train(model1, model2, seed, k=100, alpha=0.6, lr=0.002, beta2=0.99, num_epochs=150,
          batch_size=100, drop=0.5, std=0.15, fm1=16, fm2=32,
          divide_by_bs=False, w_norm=False, data_norm='pixelwise',
          early_stop=None, c=300, n_classes=10, max_epochs=80,
          max_val=30., ramp_up_mult=-5., n_samples=60000,
          print_res=True, device=None, **kwargs):
    # retrieve data
    train_dataset, test_dataset = prepare_mnist()
    ntrain = len(train_dataset)

    # build model
    model1.to(device)
    model2.to(device)

    # make data loaders
    train_loader, test_loader, indices = sample_train(train_dataset, test_dataset, batch_size,
                                                      k, n_classes, seed, shuffle_train=False)

    # setup param optimization
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, betas=(0.9, 0.99))

    # train
    model1.train()
    model2.train()
    losses = []
    sup_losses = []
    unsup_losses = []
    best_loss = 20.

