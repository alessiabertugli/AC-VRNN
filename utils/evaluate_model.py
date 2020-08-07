from utils.metrics import displacement_error, final_displacement_error, cal_l2_losses, cal_fde, cal_ade, \
    l2_loss, miss_rate, linear_velocity_acceleration_1D
from utils.absolute import relative_to_abs
from utils.adj_matrix import compute_adjs_distsim, compute_adjs_knnsim, compute_adjs
from utils.losses import l2_error_graph
import os
import torch
import argparse
import random
import numpy as np
from dataset_processing.dataset_loader import data_loader
from dataset_processing.dataloader_sdd import data_loader_sdd
from dataset_processing.dataloader_sways import data_loader_sways
from attrdict import AttrDict
from models.vrnn.vrnn_model import VRNN
from models.graph.graph_vrnn_model import GraphVRNN


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate_helper_l2(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error, 0)
        sum_ += _error[0]
    return sum_


def evaluate_baseline(args, loader, model, num_samples):
    ade_outer, fde_outer, miss_rate_outer, mean_l2_outer, best_l2_outer, max_l2_outer = [], [], [], [], [], []
    total_traj = 0
    threshold = 3
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, maps, dnames) = batch

            ade, fde, l2, losses = [], [], [], []
            total_traj += pred_traj_gt.size(1)

            for idx in range(num_samples):
                if args.model == 'vrnn':
                    kld_loss, nll_loss, _, h = model(obs_traj_rel.cuda(), obs_traj[0])
                    loss = kld_loss + nll_loss
                elif args.model == 'rnn':
                    loss, _, h = model(obs_traj_rel.cuda())

                sample_traj_rel = model.sample(args.pred_len, obs_traj_rel.size(1), obs_traj[-1], dnames, h)
                sample_traj = relative_to_abs(sample_traj_rel, obs_traj[-1])
                ade.append(displacement_error(sample_traj, pred_traj_gt.cpu(), mode='raw'))
                fde.append(final_displacement_error(sample_traj[-1], pred_traj_gt[-1].cpu(), mode='raw'))
                l2.append(l2_loss(relative_to_abs(sample_traj, obs_traj[-1]), pred_traj_gt.cpu(), loss_mask[:, args.obs_len:]))
                losses.append(loss)

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            miss_rate_outer.append(miss_rate(losses, threshold))
            mean_l2_outer.append(torch.mean(torch.stack(l2)))
            best_l2_outer.append(torch.max(torch.stack(l2)))
            max_l2_outer.append(torch.min(torch.stack(l2)))

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj
        m_rate = sum(miss_rate_outer) / total_traj
        mean_l2 = sum(mean_l2_outer) / total_traj
        best_l2 = sum(best_l2_outer) / total_traj
        max_l2 = sum(max_l2_outer) / total_traj

    return ade, fde, m_rate, mean_l2, best_l2, max_l2


def check_accuracy_baseline(args, loader, model, limit=False):
    losses = []
    metrics = {}
    val_loss = 0
    l2_losses_abs, l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, maps, dnames) = batch

            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            if args.model == 'vrnn':
                kld_loss, nll_loss, _, h = model(obs_traj_rel.cuda(), obs_traj[0])
                loss = kld_loss + nll_loss
            elif args.model == 'rnn':
                loss, _, h = model(obs_traj_rel.cuda())

            val_loss += loss.item()

            pred_traj_rel = model.sample(args.pred_len, obs_traj_rel.size(1), obs_traj[-1], dnames, h)
            pred_traj = relative_to_abs(pred_traj_rel, obs_traj[-1])

            l2_loss_abs, l2_loss_rel = cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj, pred_traj_rel, loss_mask)
            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred_traj, linear_ped, non_linear_ped)
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred_traj, linear_ped, non_linear_ped)

            losses.append(loss.item())
            l2_losses_abs.append(l2_loss_abs.item())
            l2_losses_rel.append(l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['loss'] = sum(losses) / len(losses)
    metrics['l2_loss_abs'] = sum(l2_losses_abs) / loss_mask_sum
    metrics['l2_loss_rel'] = sum(l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    model.train()
    return metrics, val_loss/len(loader)


def evaluate_graph(args, loader, model, num_samples, epoch):
    ade_outer, fde_outer, miss_rate_outer, mean_l2_outer, best_l2_outer, max_l2_outer = [], [], [], [], [], []
    mean_l2_graph = []
    total_traj = 0
    threshold = 3
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, maps, dnames) = batch

            if args.adj_type == 0:
                adj_out = compute_adjs(args, seq_start_end)
            elif args.adj_type == 1:
                adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj, pred_traj_gt)
            elif args.adj_type == 2:
                adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj, pred_traj_gt)

            ade, fde, l2, losses = [], [], [], []
            l2_graph = []
            total_traj += pred_traj_gt.size(1)

            kld_loss, nll_loss, kld_hm, h = model(obs_traj_rel.cuda(), adj_out.cuda(), seq_start_end.cuda(),
                                                  obs_traj[0], maps[:args.obs_len], epoch)

            for idx in range(num_samples):
                sample_traj_rel = model.sample(args.pred_len, seq_start_end.cuda(), False, maps[args.obs_len-1:],
                                               obs_traj[-1], dnames, h).cpu()
                sample_traj = relative_to_abs(sample_traj_rel, obs_traj[-1])
                ade.append(displacement_error(sample_traj, pred_traj_gt.cpu(), mode='raw'))
                fde.append(final_displacement_error(sample_traj[-1], pred_traj_gt[-1].cpu(), mode='raw'))
                l2.append(l2_loss(relative_to_abs(sample_traj, obs_traj[-1]), pred_traj_gt.cpu(), loss_mask[:, args.obs_len:]))
                loss = kld_loss + nll_loss + kld_hm
                losses.append(loss)

                l2_graph.append(l2_error_graph(sample_traj, pred_traj_gt.cpu()))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)
            l2_sum = evaluate_helper_l2(l2_graph, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            miss_rate_outer.append(miss_rate(losses, threshold))
            mean_l2_outer.append(torch.mean(torch.stack(l2)))
            best_l2_outer.append(torch.max(torch.stack(l2)))
            max_l2_outer.append(torch.min(torch.stack(l2)))

            mean_l2_graph.append(l2_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj
        m_rate = sum(miss_rate_outer) / total_traj
        mean_l2 = sum(mean_l2_outer) / total_traj
        best_l2 = sum(best_l2_outer) / total_traj
        max_l2 = sum(max_l2_outer) / total_traj

        l2_graph_steps = sum(mean_l2_graph) / total_traj

        mean_velocity1d, mean_velocity1d_v2,  mean_acceleration1d = linear_velocity_acceleration_1D(l2_graph_steps)

    return ade, fde, m_rate, mean_l2, best_l2, max_l2


def check_accuracy_graph(args, loader, model, epoch, limit=False):
    losses = []
    val_loss = 0
    metrics = {}
    l2_losses_abs, l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, maps, dnames) = batch

            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            if args.adj_type == 0:
                adj_out = compute_adjs(args, seq_start_end)
            elif args.adj_type == 1:
                adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj, pred_traj_gt)
            elif args.adj_type == 2:
                adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj, pred_traj_gt)

            kld_loss, nll_loss, kld_hm, h = model(obs_traj_rel.cuda(), adj_out.cuda(), seq_start_end.cuda(),
                                                  obs_traj[0], maps[:args.obs_len], epoch)
            loss = kld_loss + nll_loss + kld_hm
            val_loss += loss.item()
            pred_traj_rel = model.sample(args.pred_len, seq_start_end.cuda(), False, maps[args.obs_len-1:],
                                         obs_traj[-1], dnames, h).cpu()
            pred_traj = relative_to_abs(pred_traj_rel, obs_traj[-1])

            l2_loss_abs, l2_loss_rel = cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj, pred_traj_rel, loss_mask)
            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred_traj, linear_ped, non_linear_ped)
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred_traj, linear_ped, non_linear_ped)

            losses.append(loss.item())
            l2_losses_abs.append(l2_loss_abs.item())
            l2_losses_rel.append(l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['loss'] = sum(losses) / len(losses)
    metrics['l2_loss_abs'] = sum(l2_losses_abs) / loss_mask_sum
    metrics['l2_loss_rel'] = sum(l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    model.train()
    return metrics, val_loss/len(loader)


def evaluate_graph_sways(args, loader, model, num_samples, epoch):
    ade_outer, fde_outer = [], []
    total_traj = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, maps, dnames) = batch

            if args.adj_type == 0:
                adj_out = compute_adjs(args, seq_start_end)
            elif args.adj_type == 1:
                adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj, pred_traj_gt)
            elif args.adj_type == 2:
                adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj, pred_traj_gt)

            ade, fde, l2, losses = [], [], [], []
            total_traj += pred_traj_gt.size(1)
            kld_loss, nll_loss, kld_hm, h = model(obs_traj_rel.cuda(), adj_out.cuda(), seq_start_end.cuda(),
                                                  obs_traj[0], maps[:args.obs_len], epoch)
            for idx in range(num_samples):
                sample_traj_rel = model.sample(args.pred_len, seq_start_end.cuda(), False, maps[args.obs_len-1:],
                                               obs_traj[-1], dnames, h).cpu()
                sample_traj = relative_to_abs(sample_traj_rel, obs_traj[-1])
                ade.append(displacement_error(sample_traj, pred_traj_gt.cpu(), mode='raw'))
                fde.append(final_displacement_error(sample_traj[-1], pred_traj_gt[-1].cpu(), mode='raw'))
                loss = kld_loss + nll_loss + kld_hm
                losses.append(loss)

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj

    return ade, fde


def check_accuracy_graph_sways(args, loader, model, epoch, limit=False):
    losses = []
    val_loss = 0
    metrics = {}
    disp_error = []
    f_disp_error = []
    total_traj = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, maps, dnames) = batch

            if args.adj_type == 0:
                adj_out = compute_adjs(args, seq_start_end)
            elif args.adj_type == 1:
                adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj, pred_traj_gt)
            elif args.adj_type == 2:
                adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj, pred_traj_gt)

            kld_loss, nll_loss, kld_hm, h = model(obs_traj_rel.cuda(), adj_out.cuda(), seq_start_end.cuda(), obs_traj[0],
                                                  maps[:args.obs_len], epoch)
            loss = kld_loss + nll_loss + kld_hm
            val_loss += loss.item()
            pred_traj_rel = model.sample(args.pred_len, seq_start_end.cuda(), False, maps[args.obs_len-1:],
                                         obs_traj[-1], dnames, h).cpu()
            pred_traj = relative_to_abs(pred_traj_rel, obs_traj[-1])

            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred_traj, linear_ped=None, non_linear_ped=None)
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred_traj, linear_ped=None, non_linear_ped=None)

            losses.append(loss.item())
            disp_error.append(ade.item())

            f_disp_error.append(fde.item())
            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['loss'] = sum(losses) / len(losses)
    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    metrics['ade_l'] = 0
    metrics['fde_l'] = 0

    metrics['ade_nl'] = 0
    metrics['fde_nl'] = 0

    model.train()
    return metrics, val_loss/len(loader)


def min_nll_sampling_strategy(model, pred_traj_gt_rel, seq_start_end, maps, obs_traj, h, dnames):
    min_nll = 1e10
    best_nll_sample = torch.zeros(pred_traj_gt_rel.shape)
    for s in range(1000):
        sample_traj_rel, nll_loss_pred = model.sample_likelihood(args.pred_len, seq_start_end.cuda(),
                                                                 maps[args.obs_len - 1:], obs_traj[-1], h,
                                                                 dnames, pred_traj_gt_rel.cuda())
        if nll_loss_pred < min_nll:
            min_nll = nll_loss_pred
            best_nll_sample = sample_traj_rel
    return best_nll_sample


def get_model_baseline(checkpoint):
    args = AttrDict(checkpoint['args'])
    model = VRNN(x_dim=args.x_dim,
                 h_dim=args.h_dim,
                 z_dim=args.z_dim,
                 n_layers=args.n_layers,
                 writer=None)
    model.load_state_dict(checkpoint['best_state_ade'])
    model.cuda()
    model.train()
    return model


def get_model_graph(checkpoint):
    args = AttrDict(checkpoint['args'])
    model = GraphVRNN(args=args,
                 writer=None)
    model.load_state_dict(checkpoint['best_state_ade'])
    model.cuda()
    model.train()
    return model


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        model = get_model_graph(checkpoint)
        _args = AttrDict(checkpoint['args'])
        _args.test_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../trj2020/datasets/sdd_npy/test.npy')

        if _args.model == 'gat' or _args.model == 'gcn':
            _args.hmap_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../trj2020/dataset_processing/local_hm_5x5_sdd')

        _, loader = data_loader_sdd(_args, args.dset_type)
        ade, fde, m_rate, mean_l2, best_l2, max_l2 = evaluate_graph(_args, loader, model, args.num_samples, epoch=500)

        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dname, _args.pred_len, ade, fde))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_random_seed(76)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--num_samples', default=20, type=int)
    parser.add_argument('--dset_type', default='test', type=str)
    args = parser.parse_args()
    main(args)
