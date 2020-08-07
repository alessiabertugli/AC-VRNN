import torch
import numpy as np
from utils.block_diag_matrix import block_diag_irregular
from scipy.spatial import distance_matrix


def compute_adjs(args, seq_start_end):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        mat = []
        for t in range(0, args.obs_len + args.pred_len):
            interval = end - start
            mat.append(torch.from_numpy(np.ones((interval, interval))))
        adj_out.append(torch.stack(mat, 0))
    return block_diag_irregular(adj_out)


def compute_adjs_knnsim(args, seq_start_end, obs_traj, pred_traj_gt):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        obs_and_pred_traj = torch.cat((obs_traj, pred_traj_gt))
        knn_t = []
        for t in range(0, args.obs_len + args.pred_len):
            dists = distance_matrix(np.asarray(obs_and_pred_traj[t, start:end, :]),
                                    np.asarray(obs_and_pred_traj[t, start:end, :]))
            knn = np.argsort(dists, axis=1)[:, 0: min(args.top_k_neigh, dists.shape[0])]
            final_dists = []
            for i in range(dists.shape[0]):
                knni = np.zeros((dists.shape[1],))
                knni[knn[i]] = 1
                final_dists.append(knni)
            final_dists = np.stack(final_dists)
            knn_t.append(torch.from_numpy(final_dists))
        adj_out.append(torch.stack(knn_t, 0))
    return block_diag_irregular(adj_out)


def compute_adjs_distsim(args, seq_start_end, obs_traj, pred_traj_gt):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        obs_and_pred_traj = torch.cat((obs_traj, pred_traj_gt))
        sim_t = []
        for t in range(0, args.obs_len + args.pred_len):
            dists = distance_matrix(np.asarray(obs_and_pred_traj[t, start:end, :]),
                                    np.asarray(obs_and_pred_traj[t, start:end, :]))
            #sum_dist = np.sum(dists)
            #dists = np.divide(dists, sum_dist)
            sim = np.exp(-dists / args.sigma)
            sim_t.append(torch.from_numpy(sim))
        adj_out.append(torch.stack(sim_t, 0))
    return block_diag_irregular(adj_out)


def compute_adjs_knnsim_pred(top_k_neigh, seq_start_end, pred_traj):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        dists = distance_matrix(np.asarray(pred_traj[start:end, :]),
                                np.asarray(pred_traj[start:end, :]))
        knn = np.argsort(dists, axis=1)[:, 0: min(top_k_neigh, dists.shape[0])]
        final_dists = []
        for i in range(dists.shape[0]):
            knni = np.zeros((dists.shape[1],))
            knni[knn[i]] = 1
            final_dists.append(knni)
        final_dists = np.stack(final_dists)
        adj_out.append(torch.from_numpy(final_dists))
    return block_diag_irregular(adj_out)


def compute_adjs_distsim_pred(sigma, seq_start_end, pred_traj):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        dists = distance_matrix(np.asarray(pred_traj[start:end, :]),
                                np.asarray(pred_traj[start:end, :]))
        sim = np.exp(-dists / sigma)
        adj_out.append(torch.from_numpy(sim))
    return block_diag_irregular(adj_out)