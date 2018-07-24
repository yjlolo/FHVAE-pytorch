import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from scripts.train.train_loaders import load_data
from fhvae.fhvae import *

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="timit_np_fbank",
        help="dataset to use")
parser.add_argument("--alpha_dis", type=float, default=10.,
        help="discriminative objective weight")
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument("--n_patience", type=int, default=10,
        help="number of maximum consecutive non-improving epochs")
parser.add_argument("--n_steps_per_epoch", type=int, default=5000,
        help="number of training steps per epoch")
parser.add_argument("--n_print_steps", type=int, default=200,
        help="number of steps to print statistics")
args = parser.parse_args()
print(args)

tr_nseqs, tr_shape, tr_iterator, dt_iterator = load_data(args.dataset)

fhvae = FHVAE(nmu2=tr_nseqs, z1_dim=32, z2_dim=32,
              z1_hidden_dim=256, z2_hidden_dim=256, dec_hidden_dim=256)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fhvae.parameters())

for epoch in range(args.n_epochs):
    print("epoch %d" % (epoch+1))
    for i, (x, y, n) in enumerate(tr_iterator()):
        xin = Variable(torch.FloatTensor(x))
        xout = Variable(torch.FloatTensor(x))
        y = Variable(torch.IntTensor(y))
        n = Variable(torch.IntTensor(n))

        mu2, qz2_x, z2, qz1_x, z1, px_z, x_sample = fhvae(xin, xout, y)

        # priors
        pz1 = [0., np.log(1.0 ** 2).astype(np.float32)]
        pz2 = [mu2, np.log(0.5 ** 2).astype(np.float32)]
        pmu2 = [0., np.log(1.0 ** 2).astype(np.float32)]

        # variational lower bound
        log_pmu2 = torch.sum(log_gauss(mu2, pmu2[0], pmu2[1]), dim=1)
        kld_z2 = torch.sum(kld(qz2_x[0], qz2_x[1], pz2[0], pz2[1]), dim=1)
        kld_z1 = torch.sum(kld(qz1_x[0], qz1_x[1], pz1[0], pz1[1]), dim=1)
        log_px_z = torch.sum(log_gauss(xout, px_z[0], px_z[1]).view(xout.size(0), -1), dim=1)
        lb = log_px_z - kld_z1 - kld_z2 + log_pmu2 / n

        # discriminative loss
        logits = qz2_x[0].unsqueeze(1) - fhvae.mu2_lookup.weight.unsqueeze(0)
        logits = -1 * torch.pow(logits, 2) / (2 * torch.exp(pz2[1]))
        logits = torch.sum(logits, dim=-1)

        log_qy = criterion(logits, y)

        loss = - torch.mean(lb + args.alpha_dis * log_qy)