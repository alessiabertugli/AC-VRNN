from utils.losses import l2_loss, displacement_error, final_displacement_error
import numpy as np
import torch


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_sampled, pred_traj_sampled_rel, loss_mask):
    l2_loss_abs = l2_loss(pred_traj_sampled, pred_traj_gt, loss_mask, mode='sum')
    l2_loss_rel = l2_loss(pred_traj_sampled_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return l2_loss_abs, l2_loss_rel


def cal_ade(pred_traj_sampled, pred_traj_gt, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_sampled, pred_traj_gt)
    ade_l = torch.zeros(1)
    ade_nl = torch.zeros(1)
    if linear_ped is not None and non_linear_ped is not None:
        ade_l = displacement_error(pred_traj_sampled, pred_traj_gt, linear_ped)
        ade_nl = displacement_error(pred_traj_sampled, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(pred_traj_sampled, pred_traj_gt, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_sampled[-1], pred_traj_gt[-1])
    fde_l = torch.zeros(1)
    fde_nl = torch.zeros(1)
    if linear_ped is not None and non_linear_ped is not None:
        fde_l = final_displacement_error(pred_traj_sampled[-1], pred_traj_gt[-1], linear_ped)
        fde_nl = final_displacement_error(pred_traj_sampled[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl


def miss_rate(losses_list, threshold):
    count = 0
    for loss in losses_list:
        if loss >= threshold:
            count += 1

    return (100 * count) / len(losses_list)


def linear_velocity_acceleration_1D(sequence, seconds_between_frames=0.4):

    velocity = torch.zeros(sequence.shape).cuda()
    velocity[1:] = (sequence[1:] - sequence[:-1]) / seconds_between_frames
    velocity_v2 = torch.zeros(sequence.shape[0]-1).cuda()
    velocity_v2[1:] = sequence[2:]-sequence[:-2] / (2*seconds_between_frames)
    acceleration = torch.zeros(sequence.shape).cuda()
    acceleration[2:] = torch.abs((velocity[2:] - velocity[1:-1]) / seconds_between_frames)
    return velocity, velocity_v2, acceleration
