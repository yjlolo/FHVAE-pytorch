import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable

class FHVAE(nn.Module):
    def __init__(self, nmu2, z1_dim=32, z2_dim=32,
                 z1_hidden_dim=256, z2_hidden_dim=256, dec_hidden_dim=256, use_cuda=True):
        super(FHVAE, self).__init__()
        self.use_cuda = use_cuda
        self.input_dim = 321
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.z1_hidden_dim = z1_hidden_dim
        self.z2_hidden_dim = z2_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.mu2_lookup = nn.Embedding(nmu2, self.z2_dim)
        self.z1_pre_encoder = nn.LSTM(self.input_dim + self.z2_dim, self.z1_hidden_dim, num_layers=1,
                                      bidirectional=False, batch_first=True)
        self.z2_pre_encoder = nn.LSTM(self.input_dim, self.z2_hidden_dim, num_layers=1,
                                      bidirectional=False, batch_first=True)
        self.decoder = nn.LSTM(self.z1_dim + self.z2_dim, self.dec_hidden_dim, num_layers=1,
                               bidirectional=False, batch_first=True)
        self.z1_linear = nn.Linear(self.z1_hidden_dim, 2 * self.z1_dim)
        self.z2_linear = nn.Linear(self.z2_hidden_dim, 2 * self.z2_dim)
        self.dec_linear = nn.Linear(self.dec_hidden_dim, 2 * self.input_dim)

    def init_hidden(self, batch_size, hidden_dim):
        n_layers = 1
        h = Variable(torch.zeros(n_layers, batch_size, hidden_dim))
        c = Variable(torch.zeros(n_layers, batch_size, hidden_dim))
        if self.use_cuda:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    def encode(self, x, y):
        batch_size = x.size(0)
        T = x.size(1)
        mu2 = self.mu2_lookup(y)

        z2_hidden = self.init_hidden(batch_size, self.z2_hidden_dim)
        _, (z2_pre_out, _) = self.z2_pre_encoder(x, z2_hidden)
        z2_pre_out = z2_pre_out.squeeze()
        qz2_x = self.z2_linear(z2_pre_out)
        z2_mu, z2_logvar = torch.chunk(qz2_x, 2, dim=-1)
        qz2_x = [z2_mu, z2_logvar]
        z2_sample = self.reparameterize(z2_mu, z2_logvar)

        z1_hidden = self.init_hidden(batch_size, self.z1_hidden_dim)
        z2 = z2_sample.unsqueeze(1).repeat(1, T, 1)
        x_z2 = torch.cat([x, z2], dim=-1)
        _, (z1_pre_out, _) = self.z1_pre_encoder(x_z2, z1_hidden)
        z1_pre_out = z1_pre_out.squeeze()
        qz1_x = self.z1_linear(z1_pre_out)
        z1_mu, z1_logvar = torch.chunk(qz1_x, 2, dim=-1)
        qz1_x = [z1_mu, z1_logvar]
        z1_sample = self.reparameterize(z1_mu, z1_logvar)

        return mu2, qz2_x, z2_sample, qz1_x, z1_sample


    def decode(self, z1, z2, x):
        batch_size = x.size(0)
        z1_z2 = torch.cat([z1, z2], dim=-1).unsqueeze(1)
        x_hidden = self.init_hidden(batch_size, self.dec_hidden_dim)
        out, x_mu, x_logvar, x_sample = [], [], [], []

        for t in range(x.size(1)):
            out_t, x_hidden = self.decoder(z1_z2, x_hidden)
            px_z_t = self.dec_linear(out_t)
            x_mu_t, x_logvar_t = torch.chunk(px_z_t, 2, dim=-1)
            x_sample_t = self.reparameterize(x_mu_t, x_logvar_t)
            out.append(out_t)
            x_mu.append(x_mu_t)
            x_logvar.append(x_logvar_t)
            x_sample.append(x_sample_t)

        out = torch.cat(out, dim=1)
        x_mu = torch.cat(x_mu, dim=1)
        x_logvar = torch.cat(x_logvar, dim=1)
        x_sample = torch.cat(x_sample, dim=1)
        px_z = [x_mu, x_logvar]

        return out, px_z, x_sample

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, xin, xout, y):
        mu2, qz2_x, z2_sample, qz1_x, z1_sample = self.encode(xin, y)
        x_pre_out, px_z, x_sample = self.decode(z1_sample, z2_sample, xout)

        return mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample


def log_gauss(x, mu, logvar):
    log_2pi = torch.FloatTensor([np.log(2 * np.pi)]).cuda()
    return -0.5 * (log_2pi + logvar.data + torch.pow(x.data - mu.data, 2) / torch.exp(logvar.data))

def kld(p_mu, p_logvar, q_mu, q_logvar):
    return -0.5 * (1 + p_logvar.data - q_logvar.data - (torch.pow(p_mu.data - q_mu.data, 2) + torch.exp(p_logvar.data)) / torch.exp(q_logvar.data))
