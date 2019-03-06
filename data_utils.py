import os
import re
import struct

import numpy as np

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
                landmarks[LM_INDEX[landmark.rstrip()]] = np.array(xyz_vals.split()).astype(np.float32)
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
    values = np.flip(np.array(values).reshape((5, nrows * ncols)).T.reshape((nrows, ncols, 5)))
    return values

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

def process_data(path):
    lookup_table = {}
    id_pattern = re.compile(r"bs[0-9]{3}")
    uids = [f for f in os.listdir(path) if id_pattern.findall(f)]
    for uid in sorted(uids):
        uid_folder = os.path.join(path, uid)
        if not os.path.isdir(uid_folder): continue
        lookup_table[uid] = process_uid(uid_folder)
    return lookup_table

def convert_data(lookup_table):
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