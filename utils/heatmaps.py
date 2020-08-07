import torch
import numpy as np


def get_neighbourhood_lm_5(bin, delta_x, delta_y, batch=True):
    if batch:
        x_bin = bin[:, 0]
        y_bin = bin[:, 1]
    else:
        x_bin = bin[0]
        y_bin = bin[1]
    return np.stack([[x_bin - 2*delta_x + delta_x/2, y_bin - 2*delta_y + delta_y/2],
            [x_bin - delta_x + delta_x/2, y_bin - 2*delta_y + delta_y/2],
            [x_bin + delta_x/2, y_bin - 2*delta_y + delta_y/2],
            [x_bin + delta_x + delta_x/2, y_bin - 2*delta_y + delta_y/2],
            [x_bin + 2*delta_x + delta_x/2, y_bin - 2*delta_y + delta_y/2],
            [x_bin - 2*delta_x + delta_x/2, y_bin - delta_y + delta_y/2],
            [x_bin - delta_x + delta_x/2, y_bin - delta_y + delta_y/2],
            [x_bin + delta_x/2, y_bin - delta_y + delta_y/2],
            [x_bin + delta_x + delta_x/2, y_bin - delta_y + delta_y/2],
            [x_bin + 2*delta_x + delta_x/2, y_bin - delta_y + delta_y/2],
            [x_bin - 2*delta_x + delta_x/2, y_bin + delta_y/2],
            [x_bin - delta_x + delta_x/2, y_bin + delta_y/2],
            [x_bin + delta_x/2, y_bin + delta_y/2],
            [x_bin + delta_x + delta_x/2, y_bin + delta_y/2],
            [x_bin + 2*delta_x + delta_x/2, y_bin + delta_y/2],
            [x_bin - 2*delta_x + delta_x/2, y_bin + delta_y + delta_y/2],
            [x_bin - delta_x + delta_x/2, y_bin + delta_y + delta_y/2],
            [x_bin + delta_x/2, y_bin + delta_y + delta_y/2],
            [x_bin + delta_x + delta_x/2, y_bin + delta_y + delta_y/2],
            [x_bin + 2*delta_x + delta_x/2, y_bin + delta_y + delta_y/2],
            [x_bin - 2*delta_x + delta_x/2, y_bin + 2*delta_y + delta_y/2],
            [x_bin - delta_x + delta_x/2, y_bin + 2*delta_y + delta_y/2],
            [x_bin + delta_x/2, y_bin + 2*delta_y + delta_y/2],
            [x_bin + delta_x + delta_x/2, y_bin + 2*delta_y + delta_y/2],
            [x_bin + 2*delta_x + delta_x/2, y_bin + 2*delta_y + delta_y/2]])


def compute_similarity_matrix(next_p, neigh_centers, local_map):
    for idx, c in enumerate(neigh_centers):
        local_map[idx] += np.exp(-np.sqrt(np.power(next_p[0] - c[0], 2) + np.power(next_p[1] - c[1], 2)))
    return local_map


def get_similarity(sample, centers):
    sample = sample.permute(1, 2, 0)
    centers = centers.permute(0, 2, 1)
    sim_centers = []
    for c in centers:
        sim_centers.append(torch.exp(-torch.sqrt(torch.pow(sample[:, 0, :] - c[:, 0].unsqueeze(0).permute(1, 0), 2) \
                                                 + torch.pow(sample[:, 1, :] - c[:, 1].unsqueeze(0).permute(1, 0), 2))))
    return torch.stack(sim_centers)