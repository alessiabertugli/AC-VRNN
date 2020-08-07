import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
from dataset_processing.dataset_loader import data_loader
from utils.absolute import relative_to_abs
from models.vrnn.vrnn_model import VRNN
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimage


def draw_line(array, color, label):
    plt.plot(array[:, 0], array[:, 1], '-', c=color, label=label)


def tf_plot(x, meand_dec, pred_len):
    fig = plt.figure()
    n_pred = range(0, pred_len)

    draw_line(array=x, color='green', label='Input/observations')

    draw_line(array=meand_dec, color='blue', label='Output/Decoder means')

    for i, txt in enumerate(n_pred):
        plt.annotate(txt, (x[i][0], x[i][1]))
        plt.annotate(txt, (meand_dec[i][0], meand_dec[i][1]))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.legend()

    # Set chart title.
    plt.title("Observations vs prediction")
    # Set x, y label text.
    plt.xlabel("X")
    plt.ylabel("Y")

    return fig


def draw_all_trj(obs, gt, pred, args):
    fig = plt.figure()
    n_obs = range(0, args.obs_len)
    n_pred = range(args.obs_len, args.obs_len + args.pred_len)
    # Draw observations
    draw_line(array=obs, color='green', label='Observations')
    # Draw ground truth
    draw_line(array=gt, color='blue', label='Ground truth')
    # Draw predictions
    draw_line(array=pred, color='red', label='Predictions')

    for i, txt in enumerate(n_obs):
        plt.annotate(txt, (obs[i][0], obs[i][1]))

    for i, txt in enumerate(n_pred):
        plt.annotate(txt, (gt[i+1][0], gt[i][1]))
        plt.annotate(txt, (pred[i+1][0], pred[i][1]))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.legend()

    # Set chart title.
    plt.title("Ground truth vs prediction")
    # Set x, y label text.
    plt.xlabel("X")
    plt.ylabel("Y")
    return fig


def draw_all_trj_seq(obs, gt, pred, args):
    fig = plt.figure()

    for idx in range(obs.shape[1]):
        # Draw observations
        draw_line(array=obs[:, idx], color='green', label='Observations')
        # Draw ground truth
        draw_line(array=gt[:, idx], color='blue', label='Ground truth')
        # Draw predictions
        draw_line(array=pred[:, idx], color='red', label='Predictions')

    # Set chart title.
    plt.title("Ground truth vs prediction")
    # Set x, y label text.
    plt.xlabel("X")
    plt.ylabel("Y")
    return fig


def draw_ground_truth(gt, max_len):
    fig = plt.figure()
    n = range(0, max_len)
    draw_line(array=gt, color='blue', label='Ground truth')

    for i, txt in enumerate(n):
        plt.annotate(txt, (gt[i][0], gt[i][1]))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.legend()

    # Set chart title.
    plt.title("Ground truth")
    # Set x, y label text.
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.show()
    return fig
