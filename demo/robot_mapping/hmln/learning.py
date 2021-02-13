from utils import visualize_2d_potential
from RelationalGraph import *
from functions.NeuralNet import train_mod
from functions.ExpPotentials import PriorPotential, NeuralNetPotential, ExpWrapper, \
    TableFunction, CategoricalGaussianFunction,  ReLU, LinearLayer
from functions.MLNPotential import MLNPotential, or_op, neg_op
from learner.NeuralPMLE import PMLE
from demo.robot_mapping.robot_map_loader import load_data_fold, get_seg_type_distribution, get_subs_matrix


for model in range(5):
    train, _ = load_data_fold(model, '..')

    dt_seg_type = train['seg_type']
    dt_length = train['length']
    dt_depth = train['depth']
    dt_angle = train['angle']
    dt_neighbor = train['neighbor']
    dt_k_aligned = train['k_aligned']
    dt_aligned = train['aligned']
    dt_sharp_turn = train['sharp_turn']
    dt_single_aligned = train['single_aligned']
    dt_consecutive = train['consecutive']

    d_bool = Domain([False, True], continuous=False)
    d_length = Domain([0., 0.25], continuous=True)
    d_depth = Domain([-0.05, 0.05], continuous=True)
    d_angle = Domain([-1.57, 1.57], continuous=True)

    lv_s = LV(list(dt_seg_type.keys()))
    lv_t = LV(['wall', 'door', 'other'])

    seg_type = Atom(d_bool, [lv_s, lv_t], name='type')
    length = Atom(d_length, [lv_s], name='length')
    depth = Atom(d_depth, [lv_s], name='depth')
    angle = Atom(d_angle, [lv_s], name='angle')

    segs = np.array(lv_s.instances).reshape(-1, 1)

    fs = list()
    ps = list()

    p1_1 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | ~x[:, 1].astype(bool), dimension=2, w=None)
    p1_2 = MLNPotential(formula=lambda x: x[:, 0].astype(bool) | x[:, 1].astype(bool) | x[:, 2].astype(bool), dimension=3, w=None)
    types = np.array([
        ['wall', 'door'],
        ['door', 'other'],
        ['other', 'wall']
    ])
    subs = np.hstack([np.repeat(segs, 3, axis=0), np.tile(types, (len(segs), 1))])
    f1_1 = ParamF(p1_1, atoms=[seg_type('S', 'T1'), seg_type('S', 'T2')], lvs=['S', 'T1', 'T2'], subs=subs)
    f1_2 = ParamF(p1_2, atoms=[seg_type('S', 'wall'), seg_type('S', 'door'), seg_type('S', 'other')], lvs=['S'])
    fs.extend([f1_1, f1_2])

    p2_1 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_2 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_3 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_4 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_5 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_6 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_7 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_8 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    p2_9 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    subs = get_subs_matrix(dt_consecutive)
    f2_1 = ParamF(p2_1, atoms=[seg_type('S1', 'wall'), seg_type('S2', 'wall')], lvs=['S1', 'S2'], subs=subs)
    f2_2 = ParamF(p2_2, atoms=[seg_type('S1', 'wall'), seg_type('S2', 'door')], lvs=['S1', 'S2'], subs=subs)
    f2_3 = ParamF(p2_3, atoms=[seg_type('S1', 'wall'), seg_type('S2', 'other')], lvs=['S1', 'S2'], subs=subs)
    f2_4 = ParamF(p2_4, atoms=[seg_type('S1', 'door'), seg_type('S2', 'wall')], lvs=['S1', 'S2'], subs=subs)
    f2_5 = ParamF(p2_5, atoms=[seg_type('S1', 'door'), seg_type('S2', 'door')], lvs=['S1', 'S2'], subs=subs)
    f2_6 = ParamF(p2_6, atoms=[seg_type('S1', 'door'), seg_type('S2', 'other')], lvs=['S1', 'S2'], subs=subs)
    f2_7 = ParamF(p2_7, atoms=[seg_type('S1', 'other'), seg_type('S2', 'wall')], lvs=['S1', 'S2'], subs=subs)
    f2_8 = ParamF(p2_8, atoms=[seg_type('S1', 'other'), seg_type('S2', 'door')], lvs=['S1', 'S2'], subs=subs)
    f2_9 = ParamF(p2_9, atoms=[seg_type('S1', 'other'), seg_type('S2', 'other')], lvs=['S1', 'S2'], subs=subs)
    fs.extend([f2_1, f2_2, f2_3, f2_4, f2_5, f2_6, f2_7, f2_8, f2_9])
    ps.extend([p2_1, p2_2, p2_3, p2_4, p2_5, p2_6, p2_7, p2_8, p2_9])

    p3 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | x[:, 1].astype(bool), dimension=2, w=1)
    pairs = np.array(list(set(tuple(x) for x in get_subs_matrix(dt_consecutive)) & set(tuple(x) for x in get_subs_matrix(dt_aligned))))
    types = np.array(lv_t.instances).reshape(-1, 1)
    subs = np.hstack([np.repeat(pairs, 3, axis=0), np.tile(types, (len(pairs), 1))])
    f3 = ParamF(p3, atoms=[seg_type('S1', 'T'), seg_type('S2', 'T')], lvs=['S1', 'S2', 'T'], subs=subs)
    fs.append(f3)
    ps.append(p3)

    p4 = MLNPotential(formula=lambda x: ~x[:, 0].astype(bool) | ~x[:, 1].astype(bool), dimension=2, w=1)
    f4 = ParamF(p4, atoms=[seg_type('S1', 'door'), seg_type('S2', 'door')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_neighbor))
    fs.append(f4)
    ps.append(p4)

    p5 = MLNPotential(formula=lambda x: x[:, 0], dimension=1, w=1)
    f5 = ParamF(p5, atoms=[seg_type('S', 'door')], lvs=['S'], subs=np.array(list(dt_sharp_turn)).reshape(-1, 1))
    fs.append(f5)
    ps.append(p5)

    p6 = MLNPotential(formula=lambda x: x[:, 0], dimension=1, w=1)
    f6 = ParamF(p6, atoms=[seg_type('S', 'wall')], lvs=['S'], subs=np.array(list(dt_single_aligned)).reshape(-1, 1))
    fs.append(f6)
    ps.append(p6)

    p7 = MLNPotential(formula=lambda x: -(x[:, 1] - 0.0916) ** 2 * x[:, 0], dimension=2, w=1)
    f7 = ParamF(p7, atoms=[seg_type('S', 'door'), length('S')], lvs=['S'])
    fs.append(f7)
    ps.append(p7)

    p8 = MLNPotential(formula=lambda x: -(x[:, 1] - -0.0164) ** 2 * x[:, 0], dimension=2, w=1)
    f8 = ParamF(p8, atoms=[seg_type('S', 'door'), depth('S')], lvs=['S'])
    fs.append(f8)
    ps.append(p8)

    p9 = MLNPotential(formula=lambda x: -(x[:, 1] - 0.2674) ** 2 * x[:, 0], dimension=2, w=1)
    f9 = ParamF(p9, atoms=[seg_type('S', 'wall'), length('S')], lvs=['S'])
    fs.append(f9)
    ps.append(p9)

    p10 = MLNPotential(formula=lambda x: -(x[:, 1] - 0.0019) ** 2 * x[:, 0], dimension=2, w=1)
    f10 = ParamF(p10, atoms=[seg_type('S', 'wall'), depth('S')], lvs=['S'])
    fs.append(f10)
    ps.append(p10)

    p11 = MLNPotential(formula=lambda x: (x[:, 0] >= 0.0202) | ~x[:, 1].astype(bool), dimension=2, w=None)
    f11 = ParamF(p11, atoms=[length('S'), seg_type('S', 'wall')], lvs=['S'])
    fs.append(f11)

    p12 = MLNPotential(formula=lambda x: (x[:, 0] >= 0.0605) | ~x[:, 1].astype(bool), dimension=2, w=None)
    f12 = ParamF(p12, atoms=[length('S'), seg_type('S', 'door')], lvs=['S'])
    fs.append(f12)

    p13 = MLNPotential(formula=lambda x: (x[:, 0] <= 0.1227) | ~x[:, 1].astype(bool), dimension=2, w=None)
    f13 = ParamF(p13, atoms=[length('S'), seg_type('S', 'door')], lvs=['S'])
    fs.append(f13)

    p14 = MLNPotential(formula=lambda x: -np.log(1 + np.exp(0.0107 - x[:, 1])) * x[:, 0], dimension=2, w=1)
    f14 = ParamF(p14, atoms=[seg_type('S', 'other'), depth('S')], lvs=['S'])
    fs.append(f14)
    ps.append(p14)

    p15 = MLNPotential(formula=lambda x: -np.log(1 + np.exp(x[:, 1] + 0.0067)) * x[:, 0], dimension=2, w=1)
    f15 = ParamF(p15, atoms=[seg_type('S', 'other'), depth('S')], lvs=['S'])
    fs.append(f15)
    ps.append(p15)

    p16 = MLNPotential(formula=lambda x: (np.abs(x[:, 0]) <= 1.55) | x[:, 1].astype(bool), dimension=2, w=None)
    f16 = ParamF(p16, atoms=[angle('S'), seg_type('S', 'other')], lvs=['S'])
    fs.append(f16)

    p17_1 = MLNPotential(formula=lambda x: x[:, 0], dimension=1, w=1)
    p17_2 = MLNPotential(formula=lambda x: x[:, 0], dimension=1, w=1)
    p17_3 = MLNPotential(formula=lambda x: x[:, 0], dimension=1, w=1)
    f17_1 = ParamF(p17_1, atoms=[seg_type('S', 'wall')], lvs=['S'])
    f17_2 = ParamF(p17_2, atoms=[seg_type('S', 'door')], lvs=['S'])
    f17_3 = ParamF(p17_3, atoms=[seg_type('S', 'other')], lvs=['S'])
    fs.extend([f17_1, f17_2, f17_3])
    ps.extend([p17_1, p17_2, p17_3])

    rel_g = RelationalGraph(
        parametric_factors=fs
    )

    g, rvs_dict = rel_g.ground()

    print(len(g.rvs))

    data = dict()

    for key, rv in rvs_dict.items():
        if key[0] == length:
            data[rv] = [d_length.clip_value(dt_length[key[1]])]
        if key[0] == angle:
            data[rv] = [dt_angle[key[1]]]
        if key[0] == depth:
            data[rv] = [d_depth.clip_value(dt_depth[key[1]])]
        if key[0] == seg_type:
            if key[2] == 'wall' and dt_seg_type[key[1]] == 'W':
                data[rv] = [True]
            elif key[2] == 'door' and dt_seg_type[key[1]] == 'D':
                data[rv] = [True]
            elif key[2] == 'other' and dt_seg_type[key[1]] == 'O':
                data[rv] = [True]
            else:
                data[rv] = [False]

    def visualize(ps, t):
        if t % 200 == 0:
            temp = list()
            for p in ps:
                temp.append(p.w)
            print(temp)

    train_mod(True)
    leaner = PMLE(g, ps, data)
    leaner.train(
        lr=0.01,
        alpha=0.99,
        regular=0.0001,
        max_iter=3000,
        batch_iter=3,
        batch_size=1,
        rvs_selection_size=1000,
        sample_size=30,
        save_dir=f'learned_potentials/model_{model}',
        save_period=1000,
        visualize=visualize
    )
