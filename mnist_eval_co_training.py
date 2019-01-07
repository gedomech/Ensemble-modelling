import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import config
from co_training import train
from utils import GaussianNoise, savetime, save_exp


class CNN(nn.Module):
    """
    Original model from Laine et al.
    """
    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32, device=None):
        super(CNN, self).__init__()
        self.fm1 = fm1
        self.fm2 = fm2
        self.std = std
        self.gn = GaussianNoise(batch_size, std=self.std, device=device)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(self.fm2 * 7 * 7, 10)

    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


class CNNCoTrain(nn.Module):

    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32, device=None):
        super(CNNCoTrain, self).__init__()
        self.fm1 = fm1
        self.fm2 = fm2
        self.std = std
        self.gn = GaussianNoise(batch_size, std=self.std, device=device)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.conv1 = nn.Conv2d(1, self.fm1, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(self.fm1)
        self.conv2 = nn.Conv2d(self.fm1, self.fm2, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(self.fm2)
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(self.fm2 * 7 * 7, 10)

    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1_bn(self.conv1(x))))
        x = self.act(self.mp(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


# metrics
accs = []
accs_best = []
losses = []
sup_losses = []
unsup_losses = []
idxs = []

ts = savetime()
cfg = vars(config)

for i in range(cfg['n_exp']):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model1 = CNN(cfg['batch_size'], cfg['std'], device=device)
    # model2 = CNN(cfg['batch_ssize'], cfg['std'], device=device)

    model1 = CNNCoTrain(cfg['batch_size'], cfg['std'], device=device)
    model2 = CNNCoTrain(cfg['batch_size'], cfg['std'], device=device)

    seed = cfg['seeds'][i]
    acc, acc_best, l, sl, usl, indices = train(model1, model2, seed, device=device, **cfg)
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(indices)

print('saving experiment')

save_exp(ts, losses, sup_losses, unsup_losses,
         accs, accs_best, idxs, **cfg)

