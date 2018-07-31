import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scripts.train.train_loaders import load_data
from fhvae.datasets.wav_dataset import LogSpectrumDataset
from fhvae.fhvae import *

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, default="/home/ckycky3/deepest/FHVAE-pytorch/datasets/timit_processed",
        help="dataset directory")
parser.add_argument("--batch_size", type=int, default=256,
        help="batch size")
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
parser.add_argument("--n_save_steps", type=int, default=200,
        help="number of steps to save models")
parser.add_argument("--save_path", type=str, default="./results",
        help="path to save models")
parser.add_argument('--no_cuda', action='store_true', default=False,
        help='enables CUDA training')
args = parser.parse_args()
print(args)

use_cuda = not args.no_cuda and torch.cuda.is_available()

# tr_nseqs, tr_shape, tr_iterator, dt_iterator = load_data(args.dataset)
# tr_nseqs = len(os.listdir(args.dataset))

dataset = LogSpectrumDataset(args.data_dir)
# tr_nseqs = len(dataset)
tr_nseqs = dataset.nmu2

dataloader = DataLoader(dataset, batch_size=args.batch_size)

fhvae = FHVAE(nmu2=tr_nseqs, z1_dim=32, z2_dim=32,
              z1_hidden_dim=256, z2_hidden_dim=256, dec_hidden_dim=256, use_cuda=use_cuda)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fhvae.parameters())

if use_cuda:
    fhvae = fhvae.cuda()
    criterion = criterion.cuda()

current_step = 0
epoch = 0
while epoch < args.n_epochs:
    print("Epoch %d" % (epoch+1))
    # for x, y, n in tr_iterator():
    for i, data in enumerate(dataloader):
        xin = Variable(torch.FloatTensor(data['x']))
        xout = Variable(torch.FloatTensor(data['x']))
        y = Variable(torch.LongTensor(data['y']))
        n = Variable(torch.LongTensor(data['n']))
        if use_cuda:
            xin = xin.cuda()
            xout = xout.cuda()
            y = y.cuda()
            n = n.cuda()

        mu2, qz2_x, z2, qz1_x, z1, px_z, x_sample = fhvae(xin, xout, y)

        # priors
        zero = torch.FloatTensor([0])
        sigma_one = torch.FloatTensor([np.log(1.0 ** 2)])
        sigma_half =torch.FloatTensor([np.log(0.5 ** 2)])
        if use_cuda:
            zero = zero.cuda()
            sigma_one = sigma_one.cuda()
            sigma_half = sigma_half.cuda()

        pz1 = [zero, sigma_one]
        pz2 = [mu2, sigma_half]
        pmu2 = [zero, sigma_one]

        # variational lower bound
        log_pmu2 = torch.mean(log_gauss(mu2, pmu2[0], pmu2[1]), dim=1)
        kld_z2 = torch.mean(kld(qz2_x[0], qz2_x[1], pz2[0], pz2[1]), dim=1)
        kld_z1 = torch.mean(kld(qz1_x[0], qz1_x[1], pz1[0], pz1[1]), dim=1)
        log_px_z = torch.mean(log_gauss(xout, px_z[0], px_z[1]).view(xout.size(0), -1), dim=1)
        lb = log_px_z - kld_z1 - kld_z2 + log_pmu2 / n.float()

        # discriminative loss
        logits = qz2_x[0].unsqueeze(1) - fhvae.mu2_lookup.weight.unsqueeze(0)
        logits = -1 * torch.pow(logits, 2) / (2 * torch.exp(pz2[1]))
        logits = torch.mean(logits, dim=-1)

        log_qy = criterion(logits, y)

        loss = - torch.mean(lb + args.alpha_dis * log_qy)

        loss.backward()
        optimizer.step()

        current_step += 1

        if current_step % args.n_print_steps == 0:
            print("step %d loss %f lb %f log_px_z %f kld_z1 %f kld_z2 %f log_pmu2/n %f discriminative %f"
                  % (current_step, loss.data, torch.mean(lb).data,
                     torch.mean(log_px_z).data,torch.mean(kld_z1).data, torch.mean(kld_z2).data,
                     torch.mean(log_pmu2 / n.float()).data, torch.mean(log_qy).data,))

        if current_step % args.n_save_steps == 0:
            print("saving model, epoch %d, step %d" % (epoch+1, current_step))
            model_save_path = os.path.join(args.save_path, 'checkpoint_%d.pth.tar' % current_step)
            state_dict = {'model': fhvae.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'current_step': current_step}
            torch.save(state_dict, model_save_path)

        if current_step % args.n_steps_per_epoch == 0:
            epoch += 1