import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import config
from co_training import train
from utils import GaussianNoise, savetime, save_exp
torch.random.manual_seed(1)
torch.cuda.manual_seed(1)
import numpy as np
np.random.seed(1)
import argparse

class CNN(nn.Module):

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
        # if self.training:
        #     x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        # x = self.drop(x)
        x = self.fc(x)
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sup',action='store_true')
    parser.add_argument('--jsd',action='store_true')
    parser.add_argument('--adv',action='store_true')
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    # metrics
    accs = []

    ts = savetime()
    cfg = vars(config)

    for i in range(cfg['n_exp']):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model1 = CNN(cfg['batch_size'], cfg['std'], device=device)
        model2 = CNN(cfg['batch_size'], cfg['std'], device=device)

        seed = cfg['seeds'][i]
        acc= train(model1, model2, seed, device=device, **cfg, args= args)
        accs.append(acc)


    # print('saving experiment')
    #
    # records = {'accs':accs, 'cfg':cfg}
    # np.save('results.npy',records)


if __name__ == '__main__':
    main(get_args())
