import os
import h5py
import numpy as np

in_path = 'sub_activitynet_v1-3.c3d.hdf5'
out_path = 'c3d'

if not os.path.exists(out_path):
    os.mkdir(out_path)

d = h5py.File(in_path)
for key in d.keys():
    v_d = d[key]['c3d_features'][:].astype('float32')
    np.save(os.path.join(out_path, key+'.npy'), v_d)