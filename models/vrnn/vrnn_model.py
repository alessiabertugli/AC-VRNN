import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
from torch.autograd import Variable


class VRNN(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim, n_layers, writer, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.writer = writer

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.enc_mean = nn.Linear(h_dim, z_dim)

        self.enc_logvar = nn.Linear(h_dim, z_dim)  # nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.prior_mean = nn.Linear(h_dim, z_dim)

        self.prior_logvar = nn.Linear(h_dim, z_dim)  # nn.Softplus()

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())

        self.dec_logvar = nn.Linear(h_dim, x_dim)  # nn.Softplus()

        self.dec_mean = nn.Sequential(nn.Linear(self.h_dim, self.x_dim), nn.Hardtanh(min_val=-10, max_val=10))  # nn.Sigmoid()

        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        #self.l_abs = nn.Linear(self.x_dim, self.h_dim)

    def _encoder(self, phi_x_t, h):
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_logvar_t = self.enc_logvar(enc_t)
        return enc_mean_t, enc_logvar_t

    def _prior(self, h):
        prior_t = self.prior(h[-1])
        prior_mean_t = self.prior_mean(prior_t)
        prior_logvar_t = self.prior_logvar(prior_t)
        return prior_mean_t, prior_logvar_t

    def _decoder(self, phi_z_t, h):
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
        dec_mean_t = self.dec_mean(dec_t)
        dec_logvar_t = self.dec_logvar(dec_t)
        return dec_mean_t, dec_logvar_t

    def forward(self, x, obs_traj_in):
        """
        Inputs:
        - x: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        kld_loss, nll_loss = 0, 0
        x_list, mean_list = [torch.zeros(2)], [torch.zeros(2)]

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim), requires_grad=True).cuda()
        #h = self.l_abs(obs_traj_in.cuda()).unsqueeze(0)

        for t in range(1, x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder mean and logvar
            enc_mean_t, enc_logvar_t = self._encoder(phi_x_t, h)

            # prior mean and logvar
            prior_mean_t, prior_logvar_t = self._prior(h)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_logvar_t)
            phi_z_t = self.phi_z(z_t.cuda())

            # decoder
            dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, h)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_logvar_t, x[t])

            """
            self.writer.add_histogram('input_trajectory', x[t], t)
            self.writer.add_histogram('decoder_mean', dec_mean_t, t)
            """

            x_list.append(x[t][0])
            mean_list.append(dec_mean_t[0])

        return kld_loss, nll_loss, (x_list, mean_list), h

    def _generate_sample(self, h):
        # prior mean and logvar
        prior_mean_t, prior_logvar_t = self._prior(h)

        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_logvar_t)
        phi_z_t = self.phi_z(z_t.cuda())

        # decoder
        dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, h)

        #sample_t = self._reparameterized_sample(dec_mean_t, dec_logvar_t)

        return dec_mean_t, phi_z_t

    def sample(self, seq_len, batch_dim, h_prec=None):
        with torch.no_grad():
            if h_prec is None:
                h = Variable(torch.zeros(self.n_layers, 1, self.h_dim)).cuda()
                sample = torch.zeros(seq_len, self.x_dim)

                for t in range(seq_len):
                    sample_t, phi_z_t = self._generate_sample(h)
                    phi_x_t = self.phi_x(sample_t.view(1, -1).cuda())
                    sample[t] = sample_t.data
                    # recurrence
                    _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            else:
                h = h_prec
                sample = torch.zeros(seq_len, batch_dim, self.x_dim)

                for t in range(seq_len):
                    sample_t, phi_z_t = self._generate_sample(h)
                    phi_x_t = self.phi_x(sample_t.cuda())
                    sample[t] = sample_t.data
                    # recurrence
                    _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

        return sample

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, logvar):
        """Using std to sample"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).cuda()
        return mean + eps * std

    def _kld_gauss(self, mean_enc, logvar_enc, mean_prior, logvar_prior):
        """Using std to compute KLD"""
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) / (torch.exp(logvar_prior)), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def _nll_gauss(self, mean, logvar, x):
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll