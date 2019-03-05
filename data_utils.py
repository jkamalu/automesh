import os
import re
import struct

import numpy as np

def process_lm3(fname):
    landmarks = {}
    exp = re.compile(r"[0-9]+ landmarks:")
    with open(fname, "rt") as f:
        while not exp.findall(next(f)): continue
        while True:
            landmark = next(f, None)
            xyz_vals = next(f, None)
            if landmark and xyz_vals:
                landmarks[landmark.rstrip()] = np.array(xyz_vals.split()).astype(np.float32)
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
        
        # n_vals = nrows * ncols * 5
        n_vals = struct.unpack("I", f.read(4))[0]
        values = struct.unpack("{}d".format(n_vals), f.read(8 * n_vals))
                
    values = np.flip(np.array(values).reshape((5, nrows * ncols)).T.reshape((nrows, ncols, 5)))
    
    return values

def process_uid(uid_folder, extensions={".lm3", ".bnt"}):
    data = {}
    uid_filenames = os.listdir(uid_folder)
    for uid_filename in uid_filenames:
        fname, ext = os.path.splitext(uid_filename)

        if ext in extensions:
            uid, klass, code, number = fname.split("_")
            uid_file = os.path.join(uid_folder, uid_filename)
            
            if (klass, code, number) not in data:
                data[(klass, code, number)] = [None, None]
                
            if ext == ".lm3":
                data[(klass, code, number)][0] = process_lm3(uid_file)
            elif ext == ".bnt":
                data[(klass, code, number)][1] = process_bnt(uid_file)
    return data