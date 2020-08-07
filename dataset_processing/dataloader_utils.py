import numpy as np


def poly_fit(traj, traj_len, threshold):

    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """

    # Return evenly spaced numbers over a specified interval.
    # Returns num evenly spaced samples, calculated over the interval [start, stop].
    t = np.linspace(0, traj_len - 1, traj_len)  # 0-pred_len
    # Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y).
    # Returns a vector of coefficients p that minimises the squared error in the order deg, deg-1, … 0.
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def read_file_vci(_path, delim=','):
    data = []
    with open(_path, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split(delim)
            if line[2] == 'ped':
                line[2] = 0
                line = [float(i) for i in line]
            elif line[2] == 'veh':
                line[2] = 1
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

