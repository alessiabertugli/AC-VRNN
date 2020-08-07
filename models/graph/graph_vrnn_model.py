import torch, os
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from models.graph.gcn.gcn_model import GCN
from models.graph.gat.gat_model import GAT
from utils.adj_matrix import *
from utils.heatmaps import *
import matplotlib.pyplot as plt
from visualization.plot_model import draw_all_trj_seq
from utils.absolute import relative_to_abs


class GraphVRNN(nn.Module):

    def __init__(self, args, writer, bias=False):
        super(GraphVRNN, self).__init__()

        self.x_dim = args.x_dim
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.n_layers = args.n_layers
        self.h_init_type = args.h_init_type
        self.sigma = args.sigma
        self.top_k_neigh = args.top_k_neigh
        self.adj_type = args.adj_type
        self.use_hm = args.use_hm
        self.writer = writer
        self.k_samples_hm = args.k_samples_hm
        self.k_vloss = args.k_vloss
        self.hm_path = args.hm_path

        self.graph_hid = args.graph_hid
        self.model = args.model
        self.alpha = args.alpha
        self.nheads = args.nheads

        self.hm_dim = 25
        self.weight = args.weight

        self.rnn_type = args.rnn
        self.conditional = args.conditional

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU())

        # feature-extracting transformations for heatmap
        self.phi_hm = nn.Sequential(
            nn.Linear(self.hm_dim, self.h_dim),
            nn.LeakyReLU())

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.LeakyReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU())

        # encoder
        self.enc_hm = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim + self.h_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU())

        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)

        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU())

        # prior heatmap
        self.prior_hm = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.LeakyReLU())

        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)

        self.prior_logvar = nn.Linear(self.h_dim, self.z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU())

        # decoder hm
        self.dec_hm = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim + self.h_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU())

        self.dec_logvar = nn.Linear(self.h_dim, self.x_dim)

        self.dec_mean = nn.Sequential(nn.Linear(self.h_dim, self.x_dim), nn.Hardtanh(min_val=-10, max_val=10))
        # recurrence

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.h_dim + self.h_dim, self.h_dim, self.n_layers, bias)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.h_dim + self.h_dim, self.h_dim, self.n_layers, bias)

        self.reset_parameters()

        self.lg = nn.Linear(self.h_dim + self.h_dim, self.h_dim)

        if self.h_init_type == 1:
            h0 = torch.zeros(self.n_layers, self.h_dim)
            nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
            self.learned_h = nn.Parameter(h0, requires_grad=True)
        elif self.h_init_type == 2:
            self.l_abs = nn.Linear(self.x_dim, self.h_dim)

        # graph
        if self.model == 'gcn':
            self.graph = GCN(self.h_dim, self.graph_hid, self.h_dim)
        elif self.model == 'gat':
            self.graph = GAT(self.h_dim, self.graph_hid, self.h_dim, self.alpha, self.nheads)

        self.KL_div = nn.KLDivLoss(reduction='none')

    def _encoder(self, phi_x_t, h):
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_logvar_t = self.enc_logvar(enc_t)
        return enc_mean_t, enc_logvar_t

    def _encoder_hm(self, phi_x_t, h, phi_hm_t_prec):
        enc_t = self.enc_hm(torch.cat((phi_x_t, h[-1], phi_hm_t_prec), dim=1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_logvar_t = self.enc_logvar(enc_t)
        return enc_mean_t, enc_logvar_t

    def _prior(self, h):
        prior_t = self.prior(h[-1])
        prior_mean_t = self.prior_mean(prior_t)
        prior_logvar_t = self.prior_logvar(prior_t)
        return prior_mean_t, prior_logvar_t

    def _prior_hm(self, h, phi_hm_t_prec):
        prior_t = self.prior_hm(torch.cat((h[-1], phi_hm_t_prec), dim=1))
        prior_mean_t = self.prior_mean(prior_t)
        prior_logvar_t = self.prior_logvar(prior_t)
        return prior_mean_t, prior_logvar_t

    def _decoder(self, phi_z_t, h):
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
        dec_mean_t = self.dec_mean(dec_t)
        dec_logvar_t = self.dec_logvar(dec_t)
        return dec_mean_t, dec_logvar_t

    def _decoder_hm(self, phi_z_t, h, phi_hm_t_prec):
        dec_t = self.dec_hm(torch.cat((phi_z_t, h[-1], phi_hm_t_prec), 1))
        dec_mean_t = self.dec_mean(dec_t)
        dec_logvar_t = self.dec_logvar(dec_t)
        return dec_mean_t, dec_logvar_t

    def forward(self, x, adj, seq_start_end, obs_traj_in, maps, epoch):
        """
        Inputs:
        - x: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        kld_loss, nll_loss, kld_hm = 0, 0, 0

        if self.h_init_type == 0:
            h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim), requires_grad=True).cuda()
        elif self.h_init_type == 1:
            h = self.learned_h.repeat(1, x.size(1), 1)
        elif self.h_init_type == 2:
            h = self.l_abs(obs_traj_in.cuda()).unsqueeze(0)
        else:
            raise ValueError("Hidden state initialization unknown.")

        c = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim), requires_grad=True).cuda()

        start_abs_t = obs_traj_in.cuda().unsqueeze(0)
        cum_dec_mean_t = 0

        for t in range(1, x.size(0)):

            phi_x_t = self.phi_x(x[t])

            if self.use_hm:
                # Ground truth local heatmaps
                bin_origins = maps[t-1, :, 0]
                deltas = np.stack(maps[t-1, :, 1])
                deltas_x, deltas_y = deltas[:, 0], deltas[:, 1]
                # Normalize ground-truth local maps
                lmg = torch.from_numpy(np.stack(maps[t-1, :, 2])).type(torch.float32).cuda()
                lm_gt = lmg / (torch.sum(lmg, dim=[1, 2], keepdim=True) + 1e-8)

                if self.conditional:
                    phi_hm_t_prec = self.phi_hm(torch.flatten(lm_gt, start_dim=1))

                    # encoder mean and logvar
                    enc_mean_t, enc_logvar_t = self._encoder_hm(phi_x_t, h, phi_hm_t_prec)

                    # prior mean and logvar
                    prior_mean_t, prior_logvar_t = self._prior_hm(h, phi_hm_t_prec)
                else:
                    # encoder mean and logvar
                    enc_mean_t, enc_logvar_t = self._encoder(phi_x_t, h)

                    # prior mean and logvar
                    prior_mean_t, prior_logvar_t = self._prior(h)

                # sampling k_samples_hm
                enc_mean_t_s = torch.cat(self.k_samples_hm * [enc_mean_t], dim=0)
                enc_logvar_t_s = torch.cat(self.k_samples_hm * [enc_logvar_t], dim=0)
                z_t = self._reparameterized_sample(enc_mean_t_s, enc_logvar_t_s)
                phi_z_t = self.phi_z(z_t.cuda())

                # decoder
                if self.conditional:
                    dec_mean_t, dec_logvar_t = self._decoder_hm(phi_z_t, torch.cat(self.k_samples_hm * [h], dim=1), torch.cat(self.k_samples_hm * [phi_hm_t_prec]))
                else:
                    dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, torch.cat(self.k_samples_hm * [h], dim=1))

                dec_mean_t = dec_mean_t.reshape(self.k_samples_hm, x.size(1), self.x_dim)

                #find_nan(dec_mean_t, "dec_mean_t")

                dec_mean_abs_t = start_abs_t + cum_dec_mean_t + dec_mean_t

                #map_checking(bin_origins, deltas_x, deltas_y, dec_mean_abs_t, self.k_samples_hm)

                centers = torch.from_numpy(get_neighbourhood_lm_5(np.stack(bin_origins), deltas_x, deltas_y)).type(
                    torch.float32).cuda()

                # Compute similarity between samples and neighbourhood centers to populate the grid
                sims = get_similarity(dec_mean_abs_t, centers)
                # Sum along samples dimension and divide for k_samples_hm
                lm_sample = torch.sum(sims, 2).permute(1, 0) / self.k_samples_hm
                # Normalize maps to sum up at 1
                lm_sample = lm_sample / (torch.sum(lm_sample, dim=1, keepdim=True) + 1e-8)
                lm_sample_log = torch.log(lm_sample + 1e-8)

                # Mean of the k_samples_hm to compute h-refinement and loss
                phi_z_t = phi_z_t.reshape(self.k_samples_hm, x.size(1), self.h_dim).mean(0)
                dec_mean_t = dec_mean_t.reshape(self.k_samples_hm, x.size(1), self.x_dim).mean(0)
                dec_logvar_t = dec_logvar_t.reshape(self.k_samples_hm, x.size(1), self.x_dim).mean(0)
                cum_dec_mean_t += dec_mean_t

                kl_hm_t = self.KL_div(lm_sample_log, lm_gt.flatten(start_dim=1)).sum(1)

            else:
                kl_hm_t = torch.zeros(1).cuda()

                # prior mean and logvar
                prior_mean_t, prior_logvar_t = self._prior(h)

                # encoder mean and logvar
                enc_mean_t, enc_logvar_t = self._encoder(phi_x_t, h)

                # sampling and reparameterization
                z_t = self._reparameterized_sample(enc_mean_t, enc_logvar_t)
                phi_z_t = self.phi_z(z_t.cuda())

                # decoder
                dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, h)

            # recurrence
            if self.rnn_type == 'gru':
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            elif self.rnn_type == 'lstm':
                _, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

            # graph
            h_g = self.graph(h.squeeze(0), adj[t].cuda()).unsqueeze(0)
            h = self.lg(torch.cat((h, h_g), 2))

            # computing losses
            kld_loss_t = self._kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            nll_loss_t = self._nll_gauss(dec_mean_t, dec_logvar_t, x[t])

            if self.weight is None:
                hm_weight = dec_logvar_t.detach().exp().mean(1)
            else:
                hm_weight = 100*self.weight

            kld_loss += kld_loss_t
            nll_loss += nll_loss_t
            kld_hm += (hm_weight * kl_hm_t).mean()

        return kld_loss, nll_loss, kld_hm, h

    def _generate_sample(self, h):
        # prior mean and logvar
        prior_mean_t, prior_logvar_t = self._prior(h)
        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_logvar_t)
        phi_z_t = self.phi_z(z_t.cuda())
        # decoder
        dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, h)
        return dec_mean_t, phi_z_t

    def _generate_sample_hm(self, h, lm_gt):
        phi_hm_t_prec = self.phi_hm(torch.flatten(lm_gt, start_dim=1))
        prior_mean_t, prior_logvar_t = self._prior_hm(h, phi_hm_t_prec)
        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_logvar_t)
        phi_z_t = self.phi_z(z_t.cuda())
        # decoder
        dec_mean_t, dec_logvar_t = self._decoder_hm(phi_z_t, h, phi_hm_t_prec)
        return dec_mean_t, phi_z_t

    def _generate_sample_logvar(self, h):
        # prior mean and logvar
        prior_mean_t, prior_logvar_t = self._prior(h)
        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_logvar_t)
        phi_z_t = self.phi_z(z_t.cuda())
        # decoder
        dec_mean_t, dec_logvar_t = self._decoder(phi_z_t, h)
        return dec_mean_t, dec_logvar_t, phi_z_t

    def _generate_sample_hm_logvar(self, h, lm_gt):
        phi_hm_t_prec = self.phi_hm(torch.flatten(lm_gt, start_dim=1))
        prior_mean_t, prior_logvar_t = self._prior_hm(h, phi_hm_t_prec)
        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_logvar_t)
        phi_z_t = self.phi_z(z_t.cuda())
        # decoder
        dec_mean_t, dec_logvar_t = self._decoder_hm(phi_z_t, h, phi_hm_t_prec)
        return dec_mean_t, dec_logvar_t, phi_z_t


    def sample(self, seq_len, seq_start_end, is_sampling, maps, obs_traj_last, dnames, h_prec=None):
        c = Variable(torch.zeros(self.n_layers, h_prec.size(1), self.h_dim)).cuda()
        h = h_prec

        seq_start_end_b = seq_start_end
        if is_sampling:
            for k in range(1, self.k_vloss):
                seq_start_end_n = seq_start_end[-1, 1] + torch.stack((seq_start_end_b[:, 0], seq_start_end_b[:, 1]), 1)
                seq_start_end = torch.cat((seq_start_end, seq_start_end_n), 0)

        sample = []
        lm_gt = torch.zeros((obs_traj_last.shape[0], 5, 5))
        for t in range(seq_len):
            if self.use_hm and self.conditional:
                if t == 0:
                    lmg = torch.from_numpy(np.stack(maps[t, :, 2])).type(torch.float32).cuda()
                    lm_gt = lmg / (torch.sum(lmg, dim=[1, 2], keepdim=True) + 1e-8)
                sample_t, phi_z_t = self._generate_sample_hm(h, lm_gt)
            else:
                sample_t, phi_z_t = self._generate_sample(h)

            phi_x_t = self.phi_x(sample_t.cuda())
            sample.append(sample_t)

            sample_abs_t = relative_to_abs(sample_t.unsqueeze(0), obs_traj_last).squeeze(0)

            if self.use_hm and self.conditional:
                sample_abs_t_numpy = np.asarray(sample_abs_t.detach().cpu())

                lm_gt_list = []
                for idx in range(len(dnames)):
                    global_hm = np.load(self.hm_path + "/"+ dnames[idx]+"_local_hm.npy", allow_pickle=True)

                    abs_cord_np = np.asarray(global_hm[:, 0].tolist()).astype(np.float32)
                    deltas_np = np.asarray(global_hm[:, 1].tolist()).astype(np.float32)
                    lm_np = np.asarray(global_hm[:, 2].tolist()).astype(np.float32)

                    cond1 = (abs_cord_np[..., 0] < sample_abs_t_numpy[idx, 0:1]).astype(np.int)
                    cond2 = (sample_abs_t_numpy[idx, 0:1] <= abs_cord_np[..., 0] + deltas_np[..., 0]).astype(np.int)
                    cond3 = (abs_cord_np[..., 1] < sample_abs_t_numpy[idx, 1:2]).astype(np.int)
                    cond4 = (sample_abs_t_numpy[idx, 1:2] <= abs_cord_np[..., 1] + deltas_np[..., 1]).astype(np.int)
                    cond = cond1 * cond2 * cond3 * cond4

                    lm_gt_list.append((cond[..., np.newaxis, np.newaxis] * lm_np).sum(0))
                lm_gt = np.stack(lm_gt_list)
                lm_gt = torch.from_numpy(lm_gt).type(torch.float32).cuda()
                lm_gt = lm_gt / (torch.sum(lm_gt, dim=[1, 2], keepdim=True) + 1e-8)

            if self.adj_type == 0:
                adj_pred_t = torch.ones((sample_abs_t.shape[0], sample_abs_t.shape[0]))
            elif self.adj_type == 1:
                adj_pred_t = compute_adjs_distsim_pred(self.sigma, seq_start_end, sample_abs_t)
            elif self.adj_type == 2:
                adj_pred_t = compute_adjs_knnsim_pred(self.top_k_neigh, seq_start_end, sample_abs_t)

            # recurrence
            if self.rnn_type == 'gru':
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            elif self.rnn_type == 'lstm':
                _, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

            h_g = self.graph(h.squeeze(0), adj_pred_t.cuda()).unsqueeze(0)
            h = self.lg(torch.cat((h, h_g), 2))

        sample = torch.stack(sample)
        return sample

    def sample_likelihood(self, seq_len, seq_start_end, maps, obs_traj_last, h, dnames, pred_traj_gt_rel):

        sample = []
        nll_loss = 0
        lm_gt = torch.zeros((obs_traj_last.shape[0], 5, 5))
        for t in range(seq_len):
            if self.use_hm and self.conditional:
                if t == 0:
                    lmg = torch.from_numpy(np.stack(maps[t, :, 2])).type(torch.float32).cuda()
                    lm_gt = lmg / (torch.sum(lmg, dim=[1, 2], keepdim=True) + 1e-8)
                sample_t, dec_logvar_t, phi_z_t = self._generate_sample_hm_logvar(h, lm_gt)
            else:
                sample_t, dec_logvar_t, phi_z_t = self._generate_sample_logvar(h)

            phi_x_t = self.phi_x(sample_t.cuda())
            sample.append(sample_t)

            nll_loss_t = self._nll_gauss(sample_t, dec_logvar_t, pred_traj_gt_rel[t])
            nll_loss += (nll_loss_t/(sample_t.shape[0]*sample_t.shape[1]))

            sample_abs_t = relative_to_abs(sample_t.unsqueeze(0), obs_traj_last).squeeze(0)

            if self.use_hm and self.conditional:
                sample_abs_t_numpy = np.asarray(sample_abs_t.detach().cpu())
                lm_gt_list = []
                for idx in range(len(dnames)):
                    global_hm = np.load(self.hm_path + "/" + dnames[idx] + "_local_hm.npy", allow_pickle=True)

                    abs_cord_np = np.asarray(global_hm[:, 0].tolist()).astype(np.float32)
                    deltas_np = np.asarray(global_hm[:, 1].tolist()).astype(np.float32)
                    lm_np = np.asarray(global_hm[:, 2].tolist()).astype(np.float32)

                    cond1 = (abs_cord_np[..., 0] < sample_abs_t_numpy[idx, 0:1]).astype(np.int)
                    cond2 = (sample_abs_t_numpy[idx, 0:1] <= abs_cord_np[..., 0] + deltas_np[..., 0]).astype(np.int)
                    cond3 = (abs_cord_np[..., 1] < sample_abs_t_numpy[idx, 1:2]).astype(np.int)
                    cond4 = (sample_abs_t_numpy[idx, 1:2] <= abs_cord_np[..., 1] + deltas_np[..., 1]).astype(np.int)
                    cond = cond1 * cond2 * cond3 * cond4

                    lm_gt_list.append((cond[..., np.newaxis, np.newaxis] * lm_np).sum(0))
                lm_gt = np.stack(lm_gt_list)
                lm_gt = torch.from_numpy(lm_gt).type(torch.float32).cuda()
                lm_gt = lm_gt / (torch.sum(lm_gt, dim=[1, 2], keepdim=True) + 1e-8)

            if self.adj_type == 0:
                adj_pred_t = torch.ones((sample_abs_t.shape[0], sample_abs_t.shape[0]))
            elif self.adj_type == 1:
                adj_pred_t = compute_adjs_distsim_pred(self.sigma, seq_start_end, sample_abs_t)
            elif self.adj_type == 2:
                adj_pred_t = compute_adjs_knnsim_pred(self.top_k_neigh, seq_start_end, sample_abs_t)

            # recurrence
            if self.rnn_type == 'gru':
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            elif self.rnn_type == 'lstm':
                _, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

            h_g = self.graph(h.squeeze(0), adj_pred_t.cuda()).unsqueeze(0)
            h = self.lg(torch.cat((h, h_g), 2))

        sample = torch.stack(sample)
        nll_loss = nll_loss/seq_len
        return sample, nll_loss

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _reparameterized_sample(self, mean, logvar):
        """Using std to sample"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).cuda()
        return mean + eps * std

    def _kld_gauss(self, mean_enc, logvar_enc, mean_prior, logvar_prior):
        """Using std to compute KLD"""
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) / (torch.exp(logvar_prior) + 1e-5), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def _nll_gauss(self, mean, logvar, x):
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll
