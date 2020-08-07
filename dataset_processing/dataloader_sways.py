import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import socket

logger = logging.getLogger(__name__)
host_name = socket.gethostname()


def seq_collate(data):
    (obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, maps, dnames) = zip(*data)
    _len = [len(seq) for seq in obs_seq]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    obs_seq = torch.cat(obs_seq).permute(1, 0, 2)
    pred_seq = torch.cat(pred_seq).permute(1, 0, 2)
    obs_seq_rel = torch.cat(obs_seq_rel).permute(1, 0, 2)
    pred_seq_rel = torch.cat(pred_seq_rel).permute(1, 0, 2)
    seq_start_end = torch.LongTensor(seq_start_end)
    maps = np.concatenate(maps, 1)
    dnames = np.concatenate(dnames, 1)

    out = [obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, seq_start_end, maps, dnames]
    return tuple(out)


def cond_np(abs_cord_np, data, deltas_np):
    batch_size = 150
    total = len(data)
    rounds = (total // batch_size) + 1
    start = 0
    cond_list = []
    for _ in range(0, rounds):
        end = min(total, start + batch_size)

        cond1 = (abs_cord_np[..., 0] < data[start:end, :, 0])
        cond2 = (data[start:end, :, 0] <= (abs_cord_np[..., 0] + deltas_np[..., 0]))
        cond3 = (abs_cord_np[..., 1] < data[start:end, :, 1])
        cond4 = (data[start:end, :, 1] <= (abs_cord_np[..., 1] + deltas_np[..., 1]))
        cond = cond1 * cond2 * cond3 * cond4
        cond_list.append(cond)
        start += batch_size

    return np.concatenate(cond_list).astype(np.uint8)


class TrajectoryDataset(Dataset):

    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, map_path):

        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir

        self.obs_traj = torch.load(data_dir + "/obs.pt")
        self.pred_traj = torch.load(data_dir + "/pred.pt")
        self.seq_start_end = torch.load(data_dir + "/seq.pt").type(torch.int)
        self.seq_start_end = [(el[0].item(), el[1].item()) for el in list(self.seq_start_end)]
        self.obs_len = self.obs_traj.shape[1]
        self.seq = torch.cat((self.obs_traj, self.pred_traj), 1)
        seq_rel = torch.zeros(self.seq.shape)
        seq_rel[:, 1:] = self.seq[:, 1:] - self.seq[:, :-1]
        self.obs_traj_rel = seq_rel[:, :self.obs_len]
        self.pred_traj_rel = seq_rel[:, self.obs_len:]
        self.num_seq = len(self.seq_start_end)

        dname = data_dir.split('/')[-2]
        file = map_path + '/' + dname + '_local_hm.npy'
        global_hm = np.load(file, allow_pickle=True)
        abs_cord_np = np.asarray(global_hm[:, 0].tolist()).astype(np.float32)
        deltas_np = np.asarray(global_hm[:, 1].tolist()).astype(np.float32)
        lm_np = np.asarray(global_hm[:, 2].tolist()).astype(np.float32)

        maps = []
        dnames = []
        for seq in self.seq.numpy():
            cond1 = (abs_cord_np[..., 0] < seq[..., 0:1]).astype(np.int)
            cond2 = (seq[..., 0:1] <= abs_cord_np[..., 0] + deltas_np[..., 0]).astype(np.int)
            cond3 = (abs_cord_np[..., 1] < seq[..., 1:2]).astype(np.int)
            cond4 = (seq[..., 1:2] <= abs_cord_np[..., 1] + deltas_np[..., 1]).astype(np.int)
            cond = cond1 * cond2 * cond3 * cond4

            lm = (cond[..., np.newaxis, np.newaxis] * lm_np).sum(1)
            abs_cord = (cond[..., np.newaxis] * abs_cord_np).sum(1)
            deltas = (cond[..., np.newaxis] * deltas_np).sum(1)

            maps.append([tuple((a, b, c)) for a, b, c in zip(abs_cord, deltas, lm)])
            dnames.append(dname)

        maps = np.stack(maps)
        self.dnames = np.stack(dnames)
        self.maps = maps.transpose(1, 0, 2)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [self.obs_traj[start:end, :], self.pred_traj[start:end, :],
               self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
               self.maps[:, start:end, :], self.dnames[start:end]]
        return out


def data_loader_sways(args, mode):
    if mode == 'training':
        set = TrajectoryDataset(
            data_dir=args.training_data,
            map_path=args.hmap_path
        )
    elif mode == 'test':
        set = TrajectoryDataset(
            data_dir=args.test_data,
            map_path=args.hmap_path
        )
    elif mode == 'val':
        set = TrajectoryDataset(
            data_dir=args.validation_data,
            map_path=args.hmap_path
        )
    else:
        raise ValueError("Invalid mode!!!")

    loader = DataLoader(
        set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return set, loader
