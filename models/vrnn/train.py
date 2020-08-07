import torch.nn as nn
import torch.utils
import random
import torch.utils.data
from collections import defaultdict
from tensorboardX import SummaryWriter
from visualization.plot_model import *
from utils.absolute import relative_to_abs
from utils.losses import *
from utils.metrics import *
from utils.evaluate_model import evaluate_baseline, check_accuracy_baseline
import logging, sys

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
    ckpt_path = './output/{}/{}/{}/models/model_epo{}.pkl'.format(args.model, args.expname, args.dname, epoch)
    assert os.path.exists(ckpt_path), "epoch misspecified"
    print("loading model from %s..." % ckpt_path)
    return torch.load(ckpt_path)


def log_params(writer, args):
    writer.add_text('args', str(args), 0)


def train(epoch, train_loader, optimizer, model, args, writer, beta_vals):
    train_loss = 0
    loss_mask_sum = 0
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    l2_losses_abs, l2_losses_rel = ([],) * 2
    metrics = {}

    model.train()

    for batch_idx, batch in enumerate(train_loader):
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end, maps) = batch

        loss_mask = loss_mask[:, args.obs_len:]
        linear_ped = 1 - non_linear_ped

        # Forward + backward + optimize
        optimizer.zero_grad()
        model = model.to(device)
        kld_loss, nll_loss, (x_list, mean_list), h = model(obs_traj_rel.cuda(), obs_traj[0])
        mean_list[0] = mean_list[0].cuda()
        beta = beta_vals[epoch]

        v_losses = []
        if args.v_loss:
            for i in range(0, args.k_vloss):
                pred_traj_rel = model.sample(args.pred_len, obs_traj_rel.size(1), h)
                pred_traj_abs = relative_to_abs(pred_traj_rel, obs_traj[-1])
                ade_loss = displacement_error(pred_traj_abs, pred_traj_gt) / obs_traj_rel.size(1)
                v_losses.append(ade_loss)
            ade_min = min(v_losses)
            loss = beta * kld_loss + nll_loss + ade_min
        else:
            loss = beta * kld_loss + nll_loss

        loss.backward()

        # Clipping gradients
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()

        # Printing
        if batch_idx % args.print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(batch), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       kld_loss.item(),
                       nll_loss.item()))

            pred_traj_sampled_rel = model.sample(args.pred_len, obs_traj_rel.size(1), h)
            pred_traj_sampled = relative_to_abs(pred_traj_sampled_rel, obs_traj[-1])
            pred_traj_gt_rel = pred_traj_gt_rel
            pred_traj_gt = pred_traj_gt

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
            metrics['fde'] = sum(f_disp_error) / total_traj

            writer.add_scalar('ade', metrics['ade'], epoch)
            writer.add_scalar('fde', metrics['fde'], epoch)

    writer.add_scalar('loss_train', train_loss / len(train_loader.dataset), epoch)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    print(metrics)


def validate(checkpoint, epoch, model, train_loader, validation_loader, optimizer, writer):
    checkpoint['epoch'] = epoch
    # Check stats on the validation set
    logger.info('Checking stats on val ...')
    metrics_val, val_loss = check_accuracy_baseline(args, validation_loader, model)

    writer.add_scalar('loss_train', val_loss, epoch)

    logger.info('Checking stats on train ...')
    metrics_train, _ = check_accuracy_baseline(args, train_loader, model, limit=True)

    for k, v in sorted(metrics_val.items()):
        logger.info('  [val] {}: {:.3f}'.format(k, v))
        checkpoint['metrics_val'][k].append(v)
    for k, v in sorted(metrics_train.items()):
        logger.info('  [train] {}: {:.3f}'.format(k, v))
        checkpoint['metrics_train'][k].append(v)

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
    mean_kld_loss, mean_nll_loss = 0, 0
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
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end, maps) = batch

            loss_mask = loss_mask[:, args.obs_len:]
            linear_ped = 1 - non_linear_ped

            model = model.to(device)
            kld_loss, nll_loss, _, h = model(obs_traj_rel.cuda(), obs_traj[0])
            mean_kld_loss += beta * kld_loss.item()

            v_losses = []
            if args.v_loss:
                for j in range(0, args.k_vloss):
                    pred_traj_rel = model.sample(args.pred_len, obs_traj_rel.size(1), h)
                    pred_traj_abs = relative_to_abs(pred_traj_rel, obs_traj[-1])
                    ade_loss = displacement_error(pred_traj_abs, pred_traj_gt) / obs_traj_rel.size(1)
                    v_losses.append(ade_loss)
                ade_min = min(v_losses)
                mean_nll_loss += (nll_loss.item() + ade_min.item())
            else:
                mean_nll_loss += nll_loss.item()

            if i % args.print_every == 0:
                pred_traj_sampled_rel = model.sample(args.pred_len, obs_traj_rel.size(1), h)
                pred_traj_sampled = relative_to_abs(pred_traj_sampled_rel, obs_traj[-1])
                pred_traj_gt_rel = pred_traj_gt_rel
                pred_traj_gt = pred_traj_gt

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
                metrics['fde'] = sum(f_disp_error) / total_traj
                writer.add_scalar('ade', metrics['ade'], epoch)
                writer.add_scalar('fde', metrics['fde'], epoch)

        mean_kld_loss /= len(test_loader.dataset)
        mean_nll_loss /= len(test_loader.dataset)
        writer.add_scalar('loss_test', mean_kld_loss + mean_nll_loss, epoch)
        print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(mean_kld_loss, mean_nll_loss))
        print(metrics)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    print('**************ARGS**************')
    print(args)
    print('********************************')

    # manual seed
    set_random_seed(args.seed)
    plt.ion()

    writer_dir = os.path.join('./tf/{}/{}/{}'.format(args.model, args.expname, args.dname))
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)

    writer = SummaryWriter(writer_dir)
    log_params(writer, args)

    model = VRNN(args.x_dim, args.h_dim, args.z_dim, args.n_layers, writer)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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

    if args.reload_from >= 0:
        checkpoint = load_model(args.reload_from)
        model.load_state_dict(checkpoint['state'])

    # Generate uniform random array of n_epochs + 1 elements between 0 and 1
    beta_vals = np.concatenate([np.linspace(0.0, 1.0, args.epochs_warmup),
                                np.ones(args.n_epochs + 1 - args.epochs_warmup)])

    ade_list_test, fde_list_test, ade_list_val, fde_list_val = [], [], [], []
    for epoch in range(1, args.n_epochs + 1):
        # training + testing
        train(epoch, train_loader, optimizer, model, args, writer, beta_vals)
        test(epoch, test_loader, model, writer, beta_vals)

        # saving model
        if epoch % args.save_every == 0:
            checkpoint['state'] = model.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['optim_state'] = optimizer.state_dict()
            save_model(checkpoint, epoch)
            ade, fde, m_rate, mean_l2, best_l2, max_l2 = evaluate_baseline(args, test_loader, model, args.num_samples)
            ade_val, fde_val, m_rate_val, mean_l2_val, best_l2_val, max_l2_val = evaluate_baseline(args, validation_loader,
                                                                                                model, args.num_samples)
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
    training_path = os.path.join(os.path.curdir, '../../datasets/processed_data/hotel/train')
    test_path = os.path.join(os.path.curdir, '../../datasets/processed_data/hotel/test')
    validation_path = os.path.join(os.path.curdir, '../../datasets/processed_data/hotel/val')
    hmap_path = os.path.join(os.path.curdir, '../../dataset_processing/local_hm_5x5')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset_version', default='v1', type=str,
                        help='v1 = processed_data, v2 = eth_ucy_v2 (srlstm)')

    parser.add_argument('--model', default='vrnn', type=str)

    parser.add_argument('--training_data', default=training_path, type=str)
    parser.add_argument('--test_data', default=test_path, type=str)
    parser.add_argument('--validation_data', default=validation_path, type=str)
    parser.add_argument('--x_dim', default=2, type=int)  # tuple dimension
    parser.add_argument('--h_dim', default=64, type=int)
    parser.add_argument('--z_dim', default=16, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--n_epochs', default=500, type=int)
    parser.add_argument('--clip', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=16, type=int) #64
    parser.add_argument('--seed', default=128, type=int)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--save_every', default=25, type=int)
    parser.add_argument('--delim', default='\t')
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--epochs_warmup', default=50, type=int)
    # Variety loss params
    parser.add_argument('--v_loss', default=False, type=bool)
    parser.add_argument('--k_vloss', default=20, type=int)

    # Frequency map
    parser.add_argument('--hmap_path', default=hmap_path, type=str)

    # Metrics
    parser.add_argument('--num_samples', default=20, type=int)

    parser.add_argument('--id_optim', default=None, type=int)

    parser.add_argument('--expname', type=str, default='vl_20',
                        help='experiment name, for distinguishing different parameter settings')
    parser.add_argument('--dname', type=str, default='eth')
    parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained epoch')
    parser.add_argument('--num_samples_check', default=5000, type=int)

    args = parser.parse_args()

    main(args)