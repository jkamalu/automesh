import os
import re
import struct

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

LANDMARKS = [
    "Outer left eyebrow",
    "Middle left eyebrow",
    "Inner left eyebrow",
    "Inner right eyebrow",
    "Middle right eyebrow",
    "Outer right eyebrow",
    "Outer left eye corner",
    "Inner left eye corner",
    "Inner right eye corner",
    "Outer right eye corner",
    "Nose saddle left",
    "Nose saddle right",
    "Left nose peak",
    "Nose tip",
    "Right nose peak",
    "Left mouth corner",
    "Upper lip outer middle",
    "Right mouth corner",
    "Upper lip inner middle",
    "Lower lip inner middle",
    "Lower lip outer middle",
    "Chin middle"
]
LM_INDEX = {landmark: i for i, landmark in enumerate(LANDMARKS)}

def process_lm3(fname):
    landmarks = np.empty((len(LANDMARKS), 3))
    landmarks[:] = np.nan
    exp = re.compile(r"[0-9]+ landmarks:")
    with open(fname, "rt") as f:
        while not exp.findall(next(f)): continue
        while True:
            landmark = next(f, None)
            xyz_vals = next(f, None)
            if landmark and xyz_vals:
                try:
                    landmarks[LM_INDEX[landmark.rstrip()]] = np.array(xyz_vals.split()).astype(np.float32)
                except KeyError:
                    # rare ear and temple features
                    continue
            else:
                break
    return landmarks

def process_bnt(fname):
    with open(fname, "rb") as f:
        nrows = struct.unpack("H", f.read(2))[0]
        ncols = struct.unpack("H", f.read(2))[0]
        zmin = struct.unpack("d", f.read(8))[0]
        n_char = struct.unpack("H", f.read(2))[0]
        imfile = struct.unpack("{}c".format(n_char), f.read(1 * n_char))
        n_vals = struct.unpack("I", f.read(4))[0]
        values = struct.unpack("{}d".format(n_vals), f.read(8 * n_vals))   
    values = np.array(values).reshape((5, nrows * ncols)).T.reshape((nrows, ncols, 5))
    return np.flipud(values)

def process_uid(uid_folder, E={"lm3", "bnt"}, C={"YR", "PR", "CR", "O"}):
    data = {}
    delimiter = re.compile(r"[\._]")
    fname_splits = map(
        lambda x: tuple(x[:-1]),
        filter(
            lambda split: split[-1] in E and split[1] not in C, 
            map(
                delimiter.split,
                os.listdir(uid_folder)
            )
        )
    )    
    for split in set(fname_splits):
        uid, klass, code, number = split
        base = os.path.join(uid_folder, "_".join(split))
        data[(klass, code, number)] = []
        data[(klass, code, number)].append(process_bnt("{}.bnt".format(base)))
        data[(klass, code, number)].append(process_lm3("{}.lm3".format(base)))
        assert not np.all(np.isnan(data[(klass, code, number)][-1]))
    return data

def load_data(path):
    print("Loading data")
    lookup_table = {}
    id_pattern = re.compile(r"bs[0-9]{3}")
    uids = [f for f in os.listdir(path) if id_pattern.findall(f)]
    for i, uid in enumerate(sorted(uids)):
        if (i - 1) % 10 == 0 and i > 10:
            print("Processed data for {} users".format(i - 1))
        uid_folder = os.path.join(path, uid)
        if not os.path.isdir(uid_folder): continue
        lookup_table[uid] = process_uid(uid_folder)
    return lookup_table

def convert_data(lookup_table):
    print("Converting data")
    n_samples = 0
    landmarks = []
    face_data = []
    for uid in lookup_table:
        for config in lookup_table[uid]:
            bnt, lm3 = lookup_table[uid][config]
            landmarks.append(lm3)
            face_data.append(bnt)
            lookup_table[uid][config] = n_samples
            n_samples += 1
    return lookup_table, landmarks, face_data


def analyze_shape(face_data, h_cut, w_cut):
    shapes = np.array([d.shape[0:2] for d in face_data])

    h_cut = 280
    w_cut = 210

    h_good = shapes[:, 0][shapes[:, 0] < h_cut]
    w_good = shapes[:, 1][shapes[:, 1] < w_cut]
    s_good = shapes[np.all(np.stack([shapes[:, 0] < h_cut, shapes[:, 1] < w_cut]), axis=0)]

    print("{} ({}) at {} h_cut".format(h_good.shape[0], h_good.shape[0] / shapes.shape[0], h_cut))
    print("{} ({}) at {} w_cut".format(w_good.shape[0], w_good.shape[0] / shapes.shape[0], w_cut))
    print("{} ({}) at {} h_cut and {} w_cut".format(s_good.shape[0], s_good.shape[0] / shapes.shape[0], h_cut, w_cut))

def trim_data(lookup_table, landmarks, face_data, h_cut, w_cut):
    print("Trimming data")
    D = {}
    L = []
    F = []
    count = 0
    for uid in lookup_table:
        D[uid] = {}
        for config in lookup_table[uid]:
            idx = lookup_table[uid][config]
            if face_data[idx].shape[0] < h_cut and face_data[idx].shape[1] < w_cut:
                L.append(landmarks[idx])
                F.append(face_data[idx])
                D[uid][config] = count
                count += 1
    return D, L, F

def padding_data(face_data):
    print("Padding data")
    w_max = max(x.shape[1] for x in face_data)
    h_max = max(x.shape[0] for x in face_data)
    z_min = min(np.min(x) for x in face_data)

    for i, f in enumerate(face_data):
        w_pad = (w_max - f.shape[1]) // 2
        h_pad = (h_max - f.shape[0]) // 2
        f_pad = ((h_pad, h_max - f.shape[0] - h_pad), (w_pad, w_max - f.shape[1] - w_pad), (0, 0))
        face_data[i] = np.pad(f, f_pad, mode="constant", constant_values=z_min)
    
    return face_data

def scaling_data(face_data):
    """
    Assumes (h.3D, w.3D, z.3D, h.2D, w.2D) channel ordering
    """
    print("Scaling data")
    N, W, H, channels = face_data.shape

    for i in range(face_data.shape[0]):
        lowest, low = np.sort(np.unique(face_data[i, :, :, 2]))[:2]
        face_data[i, :, :, 2][face_data[i, :, :, 2] == lowest] = low

    scaler = MinMaxScaler()
    scaler.fit(face_data[:, :, :, 2].reshape(-1, 1))
    scaled_z = scaler.transform(face_data[:, :, :, 2].reshape(-1, 1)).reshape((N, W, H))
    face_data[:, :, :, 2] = scaled_z
    return scaler, face_data

def visualize_z(face_data, z_channel=0, lookup_table=None, uid="bs000"):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})

    if lookup_table:
        for ax, config in zip(axs.flat, lookup_table[uid]):
            idx = lookup_table[uid][config]
            ax.imshow(face_data[idx, :, :, z_channel])
            ax.set_title("_".join(config))
    else:
        for ax, datum in zip(axs.flat, face_data):
            ax.imshow(datum[:, :, z_channel])

    plt.tight_layout()
    plt.show()