import os
import math
import numpy as np


def split(path):
    all_files = os.listdir(path)
    all_files = [os.path.join(path, _path) for _path in all_files]

    file_data = []
    for file in all_files:
        dname = file.split("/")[-1].split(".")[0]
        with open(file, 'r') as f:
            lines = []
            for line in f:
                line = line.strip().split(" ")
                line = [int(line[0]), int(line[1]), float(line[2]), float(line[3])]
                lines.append(np.asarray(line))
            file_data.append((dname, np.stack(lines)))

    training_split, test_split, validation_split, all_data = [], [], [], []
    for file in file_data:
        dname = file[0]
        data = file[1]
        data_per_file = []
        # Frame IDs of the frames in the current dataset
        frame_list = np.unique(data[:, 0]).tolist()

        for frame in frame_list:
            # Extract all pedestrians in current frame
            ped_in_frame = data[data[:, 0] == frame, :]
            data_per_file.append(ped_in_frame)

        all_data.append((dname, np.concatenate(data_per_file)))
        training_split.append((dname, np.concatenate(data_per_file[: math.ceil(len(data_per_file)*0.7)])))
        validation_test_split = data_per_file[math.ceil(len(data_per_file)*0.7):]
        validation_split.append((dname, np.concatenate(validation_test_split[: math.ceil(len(data_per_file)*0.1)])))
        test_split.append((dname, np.concatenate(validation_test_split[math.ceil(len(data_per_file)*0.1):])))

    np.save(path+"../sdd_npy_videos/all_data.npy", np.asarray(all_data))
    np.save(path+"../sdd_npy_videos/training.npy", np.asarray(training_split))
    np.save(path+"../sdd_npy_videos/test.npy", np.asarray(test_split))
    np.save(path+"../sdd_npy_videos/validation.npy", np.asarray(validation_split))


if __name__ == '__main__':
    split(path="../datasets/sdd/")