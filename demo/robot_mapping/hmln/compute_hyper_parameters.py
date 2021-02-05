from demo.robot_mapping.robot_map_loader import load_data_fold
import numpy as np

data = load_data_fold(path='..')

dt_seg_type = data['seg_type']
dt_length = data['length']
dt_depth = data['depth']
dt_angle = data['angle']

wl, wd, wa = list(), list(), list()
dl, dd, da = list(), list(), list()
ol, od, oa = list(), list(), list()

for s, t in dt_seg_type.items():
    if t == 'W':
        wl.append(dt_length[s])
        wd.append(dt_depth[s])
        wa.append(dt_angle[s])
    elif t == 'D':
        dl.append(dt_length[s])
        dd.append(dt_depth[s])
        da.append(dt_angle[s])
    elif t == 'O':
        ol.append(dt_length[s])
        od.append(dt_depth[s])
        oa.append(dt_angle[s])

print('door length', np.mean(dl))
print('door depth', np.mean(dd))
print('wall length', np.mean(wl))
print('wall depth', np.mean(wd))

print('max door depth', np.mean(dd) + 1.5 * np.std(dd))
print('min wall depth', np.mean(wd) - 1.5 * np.std(wd))
print('min door length', np.mean(dl) - 1.5 * np.std(dl))
print('max door length', np.mean(dl) + 1.5 * np.std(dl))
print('min wall length', np.min(wl))
