import torch


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    :param consider_ped: Tensor of shape (batch)
    :param mode: Can be one of sum, raw
    :return: gives the euclidean displacement error
    """

    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = torch.pow(loss, 2)

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    :param pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    :param pred_pos_gt: Tensor of shape (batch, 2). Groud truth
    :param consider_ped: Tensor of shape (batch)
    :param mode: can be sum or raw
    :return: gives the euclidean displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = torch.pow(loss, 2)

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    if mode == 'sum':
        return torch.sum(loss)
    else:
        return loss


def l2_error_graph(pred_traj, pred_traj_gt):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    :param consider_ped: Tensor of shape (batch)
    :param mode: Can be one of sum, raw
    :return: gives the euclidean displacement error
    """

    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = torch.pow(loss, 2)

    loss = loss.sum(dim=2)
    return loss


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='sum'):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    :param mode: Can be one of sum, average or raw
    :return: l2 loss depending on mode
    """

    loss = (loss_mask.cpu().unsqueeze(dim=2) * (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def l2_loss_gpu(pred_traj, pred_traj_gt, loss_mask, mode='sum'):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    :param mode: Can be one of sum, average or raw
    :return: l2 loss depending on mode
    """

    loss = (loss_mask.cuda().unsqueeze(dim=2) * (pred_traj_gt.cuda().permute(1, 0, 2) - pred_traj.cuda().permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)