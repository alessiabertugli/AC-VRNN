import os
import torch
import numpy as np

input_file = '../datasets/sways_dataset/seq_hotel/hotel-8-12.npz'
output_fold = '../datasets/sways_dataset/pt/hotel/'


if __name__ == '__main__':

    print(os.path.dirname(os.path.realpath(__file__)))

    data = np.load(input_file)
    # Data come as NxTx2 numpy nd-arrays where N is the number of trajectories,
    # T is their duration.
    dataset_obsv, dataset_pred, dataset_t, the_batches = \
        data['obsvs'], data['preds'], data['times'], data['batches']
    # 4/5 of the batches to be used for training
    train_size = max(1, (len(the_batches) * 4) // 5)
    train_batches = the_batches[:train_size]
    val_size = int(train_size*0.2)
    # Test batches are the remaining ones
    test_batches = the_batches[train_size:]
    train_size = (train_size - val_size)
    val_batches = train_batches[train_size:]
    train_batches = train_batches[:train_size]

    val_batches_n = val_batches - val_batches[0][0]
    test_batches_n = test_batches - test_batches[0][0]

    train_end = train_batches[-1][1]
    val_end = val_batches[-1][1]
    obs_traj_train, obs_traj_val, obs_traj_test = dataset_obsv[:train_end], dataset_obsv[train_end:val_end],\
                                                  dataset_obsv[val_end:]
    pred_traj_train, pred_traj_val, pred_traj_test = dataset_pred[:train_end], dataset_pred[train_end:val_end],\
                                                     dataset_pred[val_end:]
    frames_train, frames_val, frames_test = dataset_t[:train_end], dataset_t[train_end:val_end], dataset_t[val_end:]

    torch.save(torch.FloatTensor(obs_traj_train), output_fold+'train/obs.pt')
    torch.save(torch.FloatTensor(obs_traj_val), output_fold+'val/obs.pt')
    torch.save(torch.FloatTensor(obs_traj_test), output_fold+'test/obs.pt')

    torch.save(torch.FloatTensor(pred_traj_train), output_fold+'train/pred.pt')
    torch.save(torch.FloatTensor(pred_traj_val), output_fold+'val/pred.pt')
    torch.save(torch.FloatTensor(pred_traj_test), output_fold+'test/pred.pt')

    torch.save(torch.Tensor(train_batches), output_fold+'train/seq.pt')
    torch.save(torch.Tensor(val_batches_n), output_fold+'val/seq.pt')
    torch.save(torch.Tensor(test_batches_n), output_fold+'test/seq.pt')

