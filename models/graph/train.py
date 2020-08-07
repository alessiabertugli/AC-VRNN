import torch.nn as nn
import torch.utils
import torch.utils.data
import random
import numpy as np
import argparse
from models.graph.graph_vrnn_model import GraphVRNN
from collections import defaultdict
from tensorboardX import SummaryWriter
from visualization.plot_model import draw_all_trj_seq
from utils.absolute import relative_to_abs
from utils.metrics import cal_ade, cal_fde, l2_loss, displacement_error, final_displacement_error, cal_l2_losses
from utils.evaluate_model import evaluate_graph, check_accuracy_graph
import os, sys
import logging
from dataset_processing.dataset_loader import data_loader
from utils.adj_matrix import compute_adjs_distsim, compute_adjs_knnsim, compute_adjs
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def save_model(checkpoint, epoch):
    dir = './checkpoints/{}/{}/{}/'.format(args.model, args.expname, args.dname)
    if not os.path.exists(dir):
        os.makedirs(dir)
    ckpt_path = os.path.join(dir, 'model_epo{}.pkl'.format(epoch))
    print("saving model to %s..." % ckpt_path)
    torch.save(checkpoint, ckpt_path)


def save_best_model(checkpoint):
    dir = './checkpoints/{}/{}/{}/'.format(args.model, args.expname, args.dname)
    if not os.path.exists(dir):
        os.makedirs(dir)
    ckpt_path = os.path.join(dir, 'best_model.pkl')
    print("saving model to %s..." % ckpt_path)
    torch.save(checkpoint, ckpt_path)


def load_model(epoch):
    ckpt_path = './checkpoints/{}/{}/{}/model_epo{}.pkl'.format(args.model, args.expname, args.dname, epoch)
    assert os.path.exists(ckpt_path), "epoch misspecified"
    print("loading model from %s..." % ckpt_path)
    return torch.load(ckpt_path)


def log_params(writer, args):
    writer.add_text('args', str(args), 0)


def train(epoch, train_loader, optimizer, model, args, writer, beta_vals):
    train_loss = 0
    mean_kld_loss, mean_nll_loss, mean_ade_loss, mean_kld_hm = 0, 0, 0, 0
    loss_mask_sum = 0
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    l2_losses_abs, l2_losses_rel = ([],) * 2
    metrics = {}

    model.train()

    beta = beta_vals[epoch]

    for batch_idx, batch in enumerate(train_loader):
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

        # Forward + backward + optimize
        optimizer.zero_grad()

        kld_loss, nll_loss, kld_hm, h = model(obs_traj_rel.cuda(), adj_out.cuda(), seq_start_end.cuda(),
                                              obs_traj[0], maps[:args.obs_len], epoch)

        v_losses = []
        if args.v_loss:
            h_samples = torch.cat(args.k_vloss * [h], 1)
            pred_traj_rel = model.sample(args.pred_len, seq_start_end.cuda(), True, maps[args.obs_len-1:],
                                         obs_traj[-1], dnames, h_samples)
            pred_traj_rel = torch.stack(torch.chunk(pred_traj_rel, args.k_vloss, dim=1))
            for k in range(0, args.k_vloss):
                pred_traj_abs = relative_to_abs(pred_traj_rel[k], obs_traj[-1])
                ade_loss = displacement_error(pred_traj_abs, pred_traj_gt) / obs_traj_rel.size(1)
                v_losses.append(ade_loss)

            ade_min = min(v_losses).cuda()
            mean_ade_loss += ade_min.item()
            loss = beta * kld_loss + nll_loss + ade_min + kld_hm
        else:
            loss = beta * kld_loss + nll_loss + kld_hm

        mean_kld_loss += kld_loss.item()
        mean_nll_loss += nll_loss.item()
        mean_kld_hm += kld_hm.item()

        loss.backward()

        # Clipping gradients
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()

        # Printing
        if batch_idx % args.print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f} \t KLD_hm: {:.6f}'.format(
                epoch, batch_idx * len(batch), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       kld_loss.item(),
                       nll_loss.item(),
                       kld_hm.item()))

            with torch.no_grad():
                pred_traj_sampled_rel = model.sample(args.pred_len, seq_start_end.cuda(), False, maps[args.obs_len-1:],
                                                     obs_traj[-1], dnames, h).cpu()
            pred_traj_sampled = relative_to_abs(pred_traj_sampled_rel, obs_traj[-1])

            ade, ade_l, ade_nl = cal_ade(pred_traj_sampled, pred_traj_gt, linear_ped, non_linear_ped)
            fde, fde_l, fde_nl = cal_fde(pred_traj_sampled, pred_traj_gt, linear_ped, non_linear_ped)
            l2_loss_abs, l2_loss_rel = cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_sampled_rel,
                                                     pred_traj_sampled, loss_mask)

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

            # Plot samples
            # Input observations (obs_len, x_len)
            start, end = seq_start_end[0][0], seq_start_end[0][1]
            input_a = obs_traj[:, start:end, :].data
            # Ground truth (pred_len, x_len)
            gt = pred_traj_gt[:, start:end, :].data
            out_a = pred_traj_sampled[:, start:end, :].data

            gt_r = np.insert(np.asarray(gt.cpu()), 0, np.asarray(input_a[-1].unsqueeze(0).cpu()), axis=0)
            out_a_r = np.insert(np.asarray(out_a.cpu()), 0, np.asarray(input_a[-1].unsqueeze(0).cpu()), axis=0)

            img2 = draw_all_trj_seq(np.asarray(input_a.cpu()), gt_r, out_a_r, args)
            writer.add_figure('Generated_samples_in_absolute_coordinates', img2, epoch)

            metrics['l2_loss_abs'] = sum(l2_losses_abs) / loss_mask_sum
            metrics['l2_loss_rel'] = sum(l2_losses_rel) / loss_mask_sum
            metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
            metrics['ade_l'] = sum(disp_error_l) / (total_traj * args.pred_len)
            metrics['ade_nl'] = sum(disp_error_nl) / (total_traj * args.pred_len)
            metrics['fde'] = sum(f_disp_error) / total_traj
            metrics['fde_l'] = sum(f_disp_error_l) / total_traj
            metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj

            writer.add_scalar('ade', metrics['ade'], epoch)
            writer.add_scalar('fde', metrics['fde'], epoch)

    mean_kld_loss /= len(train_loader)
    mean_nll_loss /= len(train_loader)
    mean_ade_loss /= len(train_loader)
    mean_kld_hm /= len(train_loader)

    writer.add_scalar('train_mean_kld_loss', mean_kld_loss, epoch)
    writer.add_scalar('train_mean_nll_loss', mean_nll_loss, epoch)
    if args.v_loss: writer.add_scalar('train_mean_ade_loss', mean_ade_loss, epoch)
    if args.use_hm: writer.add_scalar('train_mean_kld_hm', mean_kld_hm, epoch)

    writer.add_scalar('loss_train', train_loss / len(train_loader), epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader)))
    print(metrics)


def validate(checkpoint, epoch, model, train_loader, validation_loader, optimizer, writer):
    checkpoint['epoch'] = epoch
    # Check stats on the validation set
    logger.info('Checking stats on val ...')
    metrics_val, val_loss = check_accuracy_graph(args, validation_loader, model, epoch)

    writer.add_scalar('loss_train', val_loss, epoch)

    for k, v in sorted(metrics_val.items()):
        logger.info('  [val] {}: {:.3f}'.format(k, v))
        checkpoint['metrics_val'][k].append(v)

    min_ade = min(checkpoint['metrics_val']['ade'])
    min_fde = min(checkpoint['metrics_val']['fde'])

    if metrics_val['ade'] == min_ade:
        logger.info('New low for avg_disp_error')
        checkpoint['best_epoch_ade'] = epoch
        checkpoint['best_state_ade'] = model.state_dict()

    if metrics_val['fde'] == min_fde:
        logger.info('New low for avg_fianal_disp_error')
        checkpoint['best_epoch_fde'] = epoch
        checkpoint['best_state_fde'] = model.state_dict()

    # Save another checkpoint with model weights and
    # optimizer state
    checkpoint['state'] = model.state_dict()
    checkpoint['optim_state'] = optimizer.state_dict()

    save_best_model(checkpoint)
    logger.info('Saving checkpoint')
    logger.info('Done.')
    return val_loss


def test(epoch, test_loader, model, writer, beta_vals):
    """Use test data to evaluate likelihood of the model"""
    mean_kld_loss, mean_nll_loss, mean_ade_loss, mean_kld_hm = 0, 0, 0, 0
    loss_mask_sum = 0
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    l2_losses_abs, l2_losses_rel = ([],) * 2
    metrics = {}

    model.eval()

    beta = beta_vals[epoch]

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask,
             seq_start_end, maps, dnames) = batch

            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            if args.adj_type == 0:
                adj_out = compute_adjs(args, seq_start_end)
            elif args.adj_type == 1:
                adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj, pred_traj_gt)
            elif args.adj_type == 2:
                adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj, pred_traj_gt)

            kld_loss, nll_loss, kld_hm, h = model(obs_traj_rel.cuda(), adj_out.cuda(), seq_start_end.cuda(),  obs_traj[0], maps[:args.obs_len], epoch)

            mean_kld_loss += beta * kld_loss.item()
            mean_nll_loss += nll_loss.item()
            mean_kld_hm += kld_hm.item()

            v_losses = []
            if args.v_loss:
                h_samples = torch.cat(args.k_vloss * [h], 1)
                pred_traj_rel = model.sample(args.pred_len, seq_start_end.cuda(), True, maps[args.obs_len-1:], obs_traj[-1],
                                             dnames, h_samples)
                pred_traj_rel = torch.stack(torch.chunk(pred_traj_rel, args.k_vloss, dim=1))
                for k in range(0, args.k_vloss):
                    pred_traj_abs = relative_to_abs(pred_traj_rel[k], obs_traj[-1])
                    ade_loss = displacement_error(pred_traj_abs, pred_traj_gt) / obs_traj_rel.size(1)
                    v_losses.append(ade_loss)
                ade_min = min(v_losses).cuda()
                mean_ade_loss += ade_min.item()

            if i % args.print_every == 0:
                pred_traj_sampled_rel = model.sample(args.pred_len, seq_start_end.cuda(), False, maps[args.obs_len-1:],
                                                     obs_traj[-1], dnames, h).cpu()
                pred_traj_sampled = relative_to_abs(pred_traj_sampled_rel, obs_traj[-1])

                ade, ade_l, ade_nl = cal_ade(pred_traj_sampled, pred_traj_gt, linear_ped, non_linear_ped)
                fde, fde_l, fde_nl = cal_fde(pred_traj_sampled, pred_traj_gt, linear_ped, non_linear_ped)
                l2_loss_abs, l2_loss_rel = cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_sampled_rel,
                                                         pred_traj_sampled, loss_mask)

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

                metrics['l2_loss_abs'] = sum(l2_losses_abs) / loss_mask_sum
                metrics['l2_loss_rel'] = sum(l2_losses_rel) / loss_mask_sum
                metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
                metrics['ade_l'] = sum(disp_error_l) / (total_traj * args.pred_len)
                metrics['ade_nl'] = sum(disp_error_nl) / (total_traj * args.pred_len)
                metrics['fde'] = sum(f_disp_error) / total_traj
                metrics['fde_l'] = sum(f_disp_error_l) / total_traj
                metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj

                writer.add_scalar('ade', metrics['ade'], epoch)
                writer.add_scalar('fde', metrics['fde'], epoch)

        mean_kld_loss /= len(test_loader)
        mean_nll_loss /= len(test_loader)
        mean_ade_loss /= len(test_loader)
        mean_kld_hm /= len(test_loader)

        writer.add_scalar('test_mean_kld_loss', mean_kld_loss, epoch)
        writer.add_scalar('test_mean_nll_loss', mean_nll_loss, epoch)
        if args.v_loss: writer.add_scalar('test_mean_ade_loss', mean_ade_loss, epoch)
        if args.use_hm: writer.add_scalar('test_mean_kld_hm', mean_kld_hm, epoch)
        writer.add_scalar('loss_test', mean_kld_loss + mean_nll_loss + mean_ade_loss + mean_kld_hm, epoch)

        print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f}, ADE = {:.4f}, KLD_HM = {:.4f} '
              .format(mean_kld_loss, mean_nll_loss, mean_ade_loss, mean_kld_hm))
        print(metrics)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    # manual seed
    set_random_seed(args.seed)

    writer_dir = os.path.join('./tf/{}/{}/{}'.format(args.model, args.expname, args.dname))
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)

    writer = SummaryWriter(writer_dir)
    log_params(writer, args)

    model = GraphVRNN(args, writer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-4)

    if args.lr_sched == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            threshold=1e-2,
            patience=10,
            factor=5e-1,
            verbose=True
        )
    elif args.lr_sched == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.lr_sched == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Init model + optimizer + datasets
    dset, train_loader = data_loader(args, 'training')
    tset, test_loader = data_loader(args, 'test')
    vset, validation_loader = data_loader(args, 'val')

    checkpoint = {
        'args': args.__dict__,
        'state': None,
        'optim_state': None,
        'epoch': None,
        'best_epoch_ade': None,
        'best_state_ade': None,
        'best_epoch_fde': None,
        'best_state_fde': None,
        'metrics_val': defaultdict(list),
        'metrics_train': defaultdict(list)
    }

    index = 1
    if args.reload_from >= 0:
        checkpoint = load_model(args.reload_from)
        model.load_state_dict(checkpoint['state'])
        index = checkpoint['epoch'] + 1

    # Generate uniform random array of n_epochs + 1 elements between 0 and 1
    beta_vals = np.concatenate([np.linspace(0.0, 1.0, args.epochs_warmup),
                                np.ones(args.n_epochs + 1 - args.epochs_warmup)])

    ade_list_test, fde_list_test, ade_list_val, fde_list_val = [], [], [], []
    for epoch in range(index, args.n_epochs + 1):
        # training + testing
        train(epoch, train_loader, optimizer, model, args, writer, beta_vals)
        val_loss = validate(checkpoint, epoch, model, train_loader, validation_loader, optimizer, writer)
        #val_loss = 0
        if args.lr_sched == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        elif args.lr_sched == 'ExponentialLR':
            scheduler.step(epoch)
            print("Epoch ", epoch, " learning rate: ", scheduler.get_lr())
        elif args.lr_sched == 'CosineAnnealingLR':
            scheduler.step(epoch)
            print("Epoch ", epoch, " learning rate: ", scheduler.get_lr())

        test(epoch, test_loader, model, writer, beta_vals)

        # saving model
        if epoch % args.save_every == 0:
            checkpoint['state'] = model.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['optim_state'] = optimizer.state_dict()
            save_model(checkpoint, epoch)
            ade, fde, m_rate, mean_l2, best_l2, max_l2 = evaluate_graph(args, test_loader, model, args.num_samples, epoch)
            ade_val, fde_val, m_rate_val, mean_l2_val, best_l2_val, max_l2_val = evaluate_graph(args, validation_loader, model, args.num_samples, epoch)
            ade_list_test.append((ade, epoch))
            fde_list_test.append((fde, epoch))
            ade_list_val.append((ade_val, epoch))
            fde_list_val.append((fde_val, epoch))
            print("**** TEST **** ==> Ade , fde, mean l2, best l2, max l2: ", ade, fde, m_rate, mean_l2, best_l2, max_l2)
            print("**** VALIDATION **** ==> Ade , fde, mean l2, best l2, max l2: ", ade_val, fde_val, m_rate_val, mean_l2_val, best_l2_val, max_l2_val)
            if epoch == args.n_epochs:
                print("*************Min metrics*************")
                ade_min_test, ade_min_ep_test = min(ade_list_test, key=lambda t: t[0])
                fde_min_test, fde_min_ep_test = min(fde_list_test, key=lambda t: t[0])
                print("ADE Test: ", ade_min_test.item(), " epoch ", ade_min_ep_test)
                print("FDE Test: ", fde_min_test.item(), " epoch ", fde_min_ep_test)

                ade_min_val, ade_min_ep_val = min(ade_list_val, key=lambda t: t[0])
                fde_min_val, fde_min_ep_val = min(fde_list_val, key=lambda t: t[0])
                print("ADE Validation: ", ade_min_val.item(), " epoch ", ade_min_ep_val)
                print("FDE Validation: ", fde_min_val.item(), " epoch ", fde_min_ep_val)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    training_path = os.path.join(os.path.curdir, '../../../trj2020/datasets/eth_ucy_v2/zara2/train')
    test_path = os.path.join(os.path.curdir, '../../../trj2020/datasets/eth_ucy_v2/zara2/test')
    validation_path = os.path.join(os.path.curdir, '../../../trj2020/datasets/eth_ucy_v2/zara2/val')
    hmap_path = os.path.join(os.path.curdir, '../../../trj2020/dataset_processing/local_hm_5x5')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset_version', default='v1', type=str, help='v1 = processed_data, v2 = eth_ucy_v2 (srlstm)')

    parser.add_argument('--model', default='gat', type=str, help='gat or gcn')

    # Graph args
    parser.add_argument('--graph_hid', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--adj_type', type=int, default=1, help='Type of adjacency matrix: '
                                                                '0 (all connected with 1),'
                                                                '1 (distances similarity matrix),'
                                                                '2 (knn similarity matrix).')
    # Only gat
    parser.add_argument('--nheads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    parser.add_argument('--h_init_type', type=int, default=0, help='Type of initialization of h:'
                                                                    '0 (0 initialization),'
                                                                    '1 (learned initialization),'
                                                                    '2 (first absolute coordinates initialization)')
    parser.add_argument('--sigma', type=float, default=1.2, help='simga value for similarity matrix')
    parser.add_argument('--top_k_neigh', type=int, default=3)

    # Variety loss params
    parser.add_argument('--v_loss', action='store_true', help='use variety loss')  # Default false
    parser.add_argument('--k_vloss', default=1, type=int, help='k parameters of variety loss')

    # Frequency map
    parser.add_argument('--hmap_path', default=hmap_path, type=str)
    parser.add_argument('--use_hm', action='store_true')
    parser.add_argument('--conditional', action='store_true', help='belief maps as input or not')
    parser.add_argument('--k_samples_hm', default=100, type=int, help='samples for local heatmap')
    parser.add_argument('--weight', default=None, type=float, help='kl heatmap weight, if none variance weight will be used')

    # Baseline args
    parser.add_argument('--x_dim', default=2, type=int)  # tuple dimension
    parser.add_argument('--h_dim', default=64, type=int)
    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_epochs', default=500, type=int)
    parser.add_argument('--clip', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs_warmup', default=50, type=int)
    parser.add_argument('--lr_sched', default=None, type=str, help='ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau')
    parser.add_argument('--rnn', default='gru', type=str, help='gru or lstm')

    # Training args
    parser.add_argument('--training_data', default=training_path, type=str)
    parser.add_argument('--test_data', default=test_path, type=str)
    parser.add_argument('--validation_data', default=validation_path, type=str)
    parser.add_argument('--seed', default=64, type=int)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--save_every', default=25, type=int)
    parser.add_argument('--delim', default='\t')
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--id_optim', default=None, type=int)
    parser.add_argument('--expname', type=str, default='test2',
                        help='experiment name, for distinguishing different parameter settings.')
    parser.add_argument('--dname', type=str, default='zara2')
    parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained epoch.')
    parser.add_argument('--num_samples_check', default=5000, type=int)

    # Metrics
    parser.add_argument('--num_samples', default=20, type=int)

    args = parser.parse_args()

    print('**************ARGS**************')
    print(args)
    print('********************************')

    main(args=args)
