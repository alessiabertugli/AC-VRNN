import os
import matplotlib.pyplot as plt
import numpy as np
import math
from dataset_processing.dataloader_utils import read_file
from utils.heatmaps import get_neighbourhood_lm_5, compute_similarity_matrix
from utils.absolute import relative_to_abs
from dataset_processing.dataloader_utils import poly_fit


def load_sdd(data_dir):
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in all_files]
    file_data = []
    bookstore, coupa, deathCircle, gates, hyang, nexus = [], [], [], [], [], []
    for file in all_files:
        dname = file.split("/")[-1].split("_")[0]
        with open(file, 'r') as f:
            lines = []
            for line in f:
                line = line.strip().split(" ")
                line = [int(line[0]), int(line[1]), float(line[2]), float(line[3])]
                lines.append(np.asarray(line))
            if dname == "bookstore":
                bookstore.append(np.stack(lines))
            elif dname == "coupa":
                coupa.append(np.stack(lines))
            elif dname == "deathCircle":
                deathCircle.append(np.stack(lines))
            elif dname == "gates":
                gates.append(np.stack(lines))
            elif dname == "hyang":
                hyang.append(np.stack(lines))
            elif dname == "nexus":
                nexus.append(np.stack(lines))
        file_data = [("bookstore", bookstore), ("coupa", coupa), ("deathCircle", deathCircle), ("gates", gates),
                     ("hyang", hyang), ("nexus", nexus)]
    return file_data


def compute_mean_displacement_sdd(data_dir):
    file_data = load_sdd(data_dir)
    for scene in file_data:
        d_name = scene[0]
        x_min, y_min = 1e5, 1e5
        x_max, y_max = 0, 0
        id_data = []
        for data in scene[1]:
            ids = np.unique(data[:, 1]).tolist()
            for id in ids:
                id_data.append(data[id == data[:, 1], 2:4][1:] - data[id == data[:, 1], 2:4][:-1])

            if data[:, 2].min() < x_min:
                x_min = data[:, 2].min()
            if data[:, 3].min() < y_min:
                y_min = data[:, 3].min()
            if data[:, 2].max() > x_max:
                x_max = data[:, 2].max()
            if data[:, 3].max() > y_max:
                y_max = data[:, 3].max()

        id_data = np.concatenate(id_data)
        id_disp_x_mean = np.absolute(id_data[:, 0]).mean()
        id_disp_x_dev = np.sqrt(np.power((id_data[:, 0] - id_data[:, 0].mean()), 2).sum() / len(id_data[:, 0]))
        id_disp_y_mean = np.absolute(id_data[:, 1]).mean()
        id_disp_y_dev = np.sqrt(np.power((id_data[:, 1] - id_data[:, 1].mean()), 2).sum() / len(id_data[:, 1]))

        n_bins_x = (x_max - x_min) / ((id_disp_x_mean + id_disp_x_dev) / 2)
        n_bins_y = (y_max - y_min) / ((id_disp_y_mean + id_disp_y_dev) / 2)

        #np.save("stats/" + d_name + ".npy", np.asarray((int(n_bins_x), int(n_bins_y), x_min, x_max, y_min, y_max)))

        print("****************************************************")
        print("Dataset ", d_name)
        print("Mean x:", id_disp_x_mean, " Mean y:", id_disp_y_mean, " Dev x:", id_disp_x_dev, " Dev y:", id_disp_y_dev)
        print("N bins x ", int(n_bins_x), " N bins y ", int(n_bins_y))
        print("****************************************************")


def compute_mean_displacement_eth_ucy(data_dir, delim):
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in all_files]

    for path in all_files:
        # Read each file in training folder
        data = read_file(path, delim)
        d_name = path.split('/')[-1].split('.')[0]
        ids = np.unique(data[:, 1]).tolist()
        id_data = []
        for id in ids:
            id_data.append(data[id == data[:, 1], 2:4][1:] - data[id == data[:, 1], 2:4][:-1])

        id_data = np.concatenate(id_data)
        id_disp_x_mean = np.absolute(id_data[:, 0]).mean()
        id_disp_x_dev = np.sqrt(np.power((id_data[:, 0] - id_data[:, 0].mean()), 2).sum() / len(id_data[:, 0]))
        id_disp_y_mean = np.absolute(id_data[:, 1]).mean()
        id_disp_y_dev = np.sqrt(np.power((id_data[:, 1] - id_data[:, 1].mean()), 2).sum() / len(id_data[:, 1]))

        x_min, x_max = data[:, 2].min(), data[:, 2].max()
        y_min, y_max = data[:, 3].min(), data[:, 3].max()

        n_bins_x = (x_max - x_min) / ((id_disp_x_mean + id_disp_x_dev)/2)
        n_bins_y = (y_max - y_min) / ((id_disp_y_mean + id_disp_y_dev)/2)

        #np.save("stats/"+d_name+"_v2.npy", np.asarray((n_bins_x, n_bins_y)))

        print("****************************************************")
        print("Dataset ", d_name)
        print("Mean x:", id_disp_x_mean, " Mean y:", id_disp_y_mean, " Dev x:", id_disp_x_dev, " Dev y:", id_disp_y_dev)
        print("N bins x ", int(n_bins_x), " N bins y ", int(n_bins_y))
        print("****************************************************")


def compute_mean_displacement_eth_ucy_sways():
    input_file = '../datasets/sways_dataset/st3_dataset/students-8-12.npz'
    output_fold = '../dataset_processing/stats/'
    data = np.load(input_file)
    data = np.concatenate((data['obsvs'], data['preds']), 1)
    disp_ped_seq = data[:, 1:] - data[:, :-1]
    id_disp_x_mean = np.absolute(disp_ped_seq[:, 0]).mean()
    id_disp_x_dev = np.sqrt(np.power((disp_ped_seq[:, 0] - disp_ped_seq[:, 0].mean()), 2).sum() / len(disp_ped_seq[:, 0]))
    id_disp_y_mean = np.absolute(disp_ped_seq[:, 1]).mean()
    id_disp_y_dev = np.sqrt(np.power((disp_ped_seq[:, 1] - disp_ped_seq[:, 1].mean()), 2).sum() / len(disp_ped_seq[:, 1]))

    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()

    n_bins_x = (x_max - x_min) / ((id_disp_x_mean + id_disp_x_dev) / 2)
    n_bins_y = (y_max - y_min) / ((id_disp_y_mean + id_disp_y_dev) / 2)

    np.save(output_fold+"univ_sw.npy", np.asarray((n_bins_x, n_bins_y)))

    print("****************************************************")
    print("Mean x:", id_disp_x_mean, " Mean y:", id_disp_y_mean, " Dev x:", id_disp_x_dev, " Dev y:", id_disp_y_dev)
    print("N bins x ", int(n_bins_x), " N bins y ", int(n_bins_y))
    print("****************************************************")


def get_heatmap(data, x_min, y_min, x_max, y_max, n_bins_x, n_bins_y, out_data_dir):
    delta_x = (x_max - x_min) / n_bins_x
    delta_y = (y_max - y_min) / n_bins_y
    x_bins, y_bins = [], []
    for i in range(0, n_bins_x + 1):
        x_bins.append(x_min + i * delta_x)
    for i in range(0, n_bins_y + 1):
        y_bins.append(y_min + i * delta_y)

    l = []
    [l.append((x_bins[i], y_bins[j])) for i in range(len(x_bins) - 1) for j in range(len(y_bins) - 1)]
    l = np.asarray(l)

    ids = np.unique(data[:, 1]).tolist()
    ids_data = []
    for id in ids:
        ids_data.append(data[id == data[:, 1], :])

    total_maps = []
    for bin in l:
        local_map = np.zeros(25)
        neigh_centers = get_neighbourhood_lm_5(bin, delta_x, delta_y, batch=False)
        for id in ids_data:
            for idx, point in enumerate(id[0:-1]):
                if bin[0] < point[2] <= (bin[0] + delta_x) and bin[1] < point[3] <= (bin[1] + delta_y):
                    next_p = id[idx + 1, 2:4]
                    local_map = compute_similarity_matrix(next_p, neigh_centers, local_map)
        total_maps.append(
            [np.asarray(bin), np.asarray([delta_x, delta_y]), np.asarray(np.flip(local_map.reshape(5, 5), 0))])
    np.save(out_data_dir + '_local_hm.npy', np.asarray(total_maps))


def get_heatmap_sdd(data, x_min, y_min, x_max, y_max, n_bins_x, n_bins_y, out_data_dir):
    delta_x = (x_max - x_min) / n_bins_x
    delta_y = (y_max - y_min) / n_bins_y
    x_bins, y_bins = [], []
    for i in range(0, n_bins_x + 1):
        x_bins.append(x_min + i * delta_x)
    for i in range(0, n_bins_y + 1):
        y_bins.append(y_min + i * delta_y)

    l = []
    [l.append((x_bins[i], y_bins[j])) for i in range(len(x_bins) - 1) for j in range(len(y_bins) - 1)]
    l = np.asarray(l)

    total_maps = []
    print("Datatset: ", out_data_dir.split("/")[-1])
    for idx, bin in enumerate(l):
        print("Bin ", idx, "/", len(l))
        local_map = np.zeros(25)
        neigh_centers = get_neighbourhood_lm_5(bin, delta_x, delta_y, batch=False)
        for trj in data:
            for idx, point in enumerate(trj[0:-1]):
                if bin[0] < point[2] <= (bin[0] + delta_x) and bin[1] < point[3] <= (bin[1] + delta_y):
                    next_p = trj[idx + 1, 2:4]
                    local_map = compute_similarity_matrix(next_p, neigh_centers, local_map)
        total_maps.append(
            [np.asarray(bin), np.asarray([delta_x, delta_y]), np.asarray(np.flip(local_map.reshape(5, 5), 0))])
    np.save(out_data_dir + '_local_hm.npy', np.asarray(total_maps))


# 5x5 neighbourhood
def compute_local_heatmaps_eth_ucy(data_dir, out_dir, delim):
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in all_files]
    stats_files = os.listdir("stats")
    stats_files = [os.path.join("stats", _path) for _path in stats_files]
    for path in all_files:
        dname = path.split('/')[-1].split('.')[0]

        for file in stats_files:
            st_name = file.split('/')[-1].split('.')[0]
            if st_name == dname:
                n_bins = np.load(file)
                n_bins_x, n_bins_y = int(n_bins[0]), int(n_bins[1])

        # Read each file in training folder
        data = read_file(path, delim)
        out_data_dir = out_dir+'/'+path.split('/')[-1].split('.')[0]

        x_min, x_max = data[:, 2].min(), data[:, 2].max()
        y_min, y_max = data[:, 3].min(), data[:, 3].max()

        get_heatmap(data, x_min, y_min, x_max, y_max, n_bins_x, n_bins_y, out_data_dir)


def compute_local_heatmaps_sdd(data_dir, out_dir):
    file_data = load_sdd(data_dir)

    stats_files = os.listdir("stats")
    stats_files = [os.path.join("stats", _path) for _path in stats_files]

    for data in file_data:
        dname = data[0]

        for file in stats_files:
            st_name = file.split('/')[-1].split('.')[0]
            if st_name == dname:
                n_bins_x, n_bins_y, x_min, x_max, y_min, y_max = np.load(file)

        out_data_dir = out_dir + '/'+dname
        all_ped = []
        for arr in data[1]:
            ids = np.unique(arr[:, 1]).tolist()
            for id in ids:
                all_ped.append(arr[id == arr[:, 1], :])

        get_heatmap_sdd(all_ped, x_min, y_min, x_max, y_max, int(n_bins_x), int(n_bins_y), out_data_dir)


def compute_local_hm_eth_ucy_sways(data_file, stats_file, out_dir):
    n_bins_x, n_bins_y = int(np.load(stats_file)[0]),  int(np.load(stats_file)[1])
    data = np.load(data_file)
    data = np.concatenate((data['obsvs'], data['preds']), 1)

    x_min, x_max = data[:, :, 0].min(), data[:, :, 0].max()
    y_min, y_max = data[:, :, 1].min(), data[:, :, 1].max()

    delta_x = (x_max - x_min) / n_bins_x
    delta_y = (y_max - y_min) / n_bins_y
    x_bins, y_bins = [], []
    for i in range(0, n_bins_x + 1):
        x_bins.append(x_min + i * delta_x)
    for i in range(0, n_bins_y + 1):
        y_bins.append(y_min + i * delta_y)

    l = []
    [l.append((x_bins[i], y_bins[j])) for i in range(len(x_bins) - 1) for j in range(len(y_bins) - 1)]
    l = np.asarray(l)

    total_maps = []
    for bin in l:
        local_map = np.zeros(25)
        neigh_centers = get_neighbourhood_lm_5(bin, delta_x, delta_y, batch=False)
        for seq in data:
            for idx, point in enumerate(seq[:-1]):
                if bin[0] < point[0] <= (bin[0] + delta_x) and bin[1] < point[1] <= (bin[1] + delta_y):
                    next_p = seq[idx + 1]
                    local_map = compute_similarity_matrix(next_p, neigh_centers, local_map)
        total_maps.append(
            [np.asarray(bin), np.asarray([delta_x, delta_y]), np.asarray(np.flip(local_map.reshape(5, 5), 0))])
    np.save(out_dir, np.asarray(total_maps))


if __name__ == '__main__':
    compute_mean_displacement_eth_ucy(data_dir='../datasets/path-to-eth_ucy-dataset', delim='\t')
    compute_local_heatmaps_eth_ucy(data_dir='../datasets/path-to-eth_ucy-dataset', out_dir='local_hm_5x5', delim='\t')