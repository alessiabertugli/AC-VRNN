import logging
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset_processing.dataloader_utils import read_file, poly_fit
import socket
import pickle

logger = logging.getLogger(__name__)
host_name = socket.gethostname()


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, seq_list, seq_list_rel, maps, dnames) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    maps = np.concatenate(maps, 0).transpose(2, 0, 1)
    dnames = np.concatenate(dnames, 0)

    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, seq_start_end, maps, dnames]
    return tuple(out)


class TrajectoryDataset(Dataset):

    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len, pred_len, skip, delim, mode, map_path, debug, threshold=0.002, min_ped=1):

        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """

        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        maps = []
        dnames = []

        for path in all_files:
            # Read each file in training folder
            data = read_file(path, delim)
            if mode == 'training':
                dname = path.split('/')[-1].split('.')[0].split('_train')[0]
            elif mode == 'test':
                dname = path.split('/')[-1].split('.')[0]
            elif mode == 'val':
                dname = path.split('/')[-1].split('.')[0].split('_val')[0]

            file = map_path + '/' + dname + '_local_hm.npy'
            global_hm = np.load(file, allow_pickle=True)

            abs_cord_np = np.asarray(global_hm[:, 0].tolist()).astype(np.float32)
            deltas_np = np.asarray(global_hm[:, 1].tolist()).astype(np.float32)
            lm_np = np.asarray(global_hm[:, 2].tolist()).astype(np.float32)

            abs_cord_np = np.repeat(np.expand_dims(abs_cord_np, 0), 20, axis=0)
            deltas_np = np.repeat(np.expand_dims(deltas_np, 0), 20, axis=0)
            lm_np = np.repeat(np.expand_dims(lm_np, 0), 20, axis=0)

            # Unique frame indices list for each training file
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                # List of list: contains all positions in each frame
                frame_data.append(data[frame == data[:, 0], :])
            # Num sequences to consider for a specified sequence length (in this case 20)
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))
            # Cycle for the length of sequences
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                # All data for the current sequence: from the current index to current_index + sequence length (ex 8-38)
                curr_seq_data = np.concatenate(frame_data[idx:(idx + self.seq_len)], axis=0)  # [frame_id, ped_id, x, y]
                # IDs of pedestrians in the current sequence
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))

                num_peds_considered = 0
                _non_linear_ped = []
                # Cycle on pedestrians for the current sequence index
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # Current sequence for each pedestrian
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # Start frame for the current sequence of the current pedestrian reported to 0
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    # End frame for the current sequence of the current pedestrian:
                    # end of current pedestrian path in the current sequence.
                    # It can be sequence length if the pedestrian appears in all frame of the sequence
                    # or less if it disappears earlier.
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    # Exclude trajectories less then seq_len
                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq  # [frame_id, ped_id, x, y]
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])

                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

                    for seq in curr_seq[:num_peds_considered].transpose(0, 2, 1):
                        cond1 = (abs_cord_np[..., 0] < seq[:, 0:1]).astype(np.int)
                        cond2 = (seq[:, 0:1] <= abs_cord_np[..., 0] + deltas_np[..., 0]).astype(np.int)
                        cond3 = (abs_cord_np[..., 1] < seq[..., 1:2]).astype(np.int)
                        cond4 = (seq[:, 1:2] <= abs_cord_np[..., 1] + deltas_np[..., 1]).astype(np.int)
                        cond = cond1 * cond2 * cond3 * cond4

                        lm = (cond[..., np.newaxis, np.newaxis] * lm_np).sum(1)
                        abs_cord = (cond[..., np.newaxis] * abs_cord_np).sum(1)
                        deltas = (cond[..., np.newaxis] * deltas_np).sum(1)

                        maps.append([tuple((a, b, c)) for a, b, c in zip(abs_cord, deltas, lm)])
                        dnames.append(dname)

        maps = np.stack(maps)
        self.dnames = np.stack(dnames)
        self.num_seq = len(seq_list)
        self.seq_list = np.concatenate(seq_list, axis=0)
        self.seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        self.maps = maps.transpose(0, 2, 1)
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(self.seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(self.seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.seq_list = torch.from_numpy(self.seq_list).type(torch.float)
        self.seq_list_rel = torch.from_numpy(self.seq_list_rel).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        # Cumulative indices of pedestrian
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # Start and nd indices of the pedestrians. It indicates the start and end indices of the trajectories to sample
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [self.obs_traj[start:end, :], self.pred_traj[start:end, :],
               self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
               self.non_linear_ped[start:end], self.loss_mask[start:end, :],
               self.seq_list[start:end, :], self.seq_list_rel[start:end, :],
               self.maps[start:end, :], self.dnames[start:end]]
        return out


def data_loader(args, mode):
    if mode == 'training':
        set = TrajectoryDataset(
            data_dir=args.training_data,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            delim=args.delim,
            mode=mode,
            map_path=args.hmap_path,
            debug=args.debug
        )
    elif mode == 'test':
        set = TrajectoryDataset(
            data_dir=args.test_data,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            delim=args.delim,
            mode=mode,
            map_path=args.hmap_path,
            debug=args.debug
        )
    elif mode == 'val':
        set = TrajectoryDataset(
            data_dir=args.validation_data,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            delim=args.delim,
            mode=mode,
            map_path=args.hmap_path,
            debug=args.debug
        )
    else:
        raise ValueError("Invalid mode!!!")

    loader = DataLoader(
        set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        drop_last=True)
    return set, loader
