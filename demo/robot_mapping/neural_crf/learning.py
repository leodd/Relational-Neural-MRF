from utils import visualize_2d_potential
from RelationalGraph import *
from functions.NeuralNet import train_mod
from functions.ExpPotentials import PriorPotential, NeuralNetPotential, ExpWrapper, \
    TableFunction, CategoricalGaussianFunction,  ReLU, LinearLayer
from functions.ConditionalNeuralPotentials import ConditionalNeuralPotential
from functions.MLNPotential import MLNPotential
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

    d_seg_type = Domain(['W', 'D', 'O'], continuous=False)
    d_length = Domain([0., 0.25], continuous=True)
    d_depth = Domain([-0.05, 0.05], continuous=True)
    d_angle = Domain([-1.57, 1.57], continuous=True)
    d_seg_type.domain_indexize()

    lv_s = LV(list(dt_seg_type.keys()))

    seg_type = Atom(d_seg_type, [lv_s], name='type')
    length = Atom(d_length, [lv_s], name='length')
    depth = Atom(d_depth, [lv_s], name='depth')
    angle = Atom(d_angle, [lv_s], name='angle')

    p_lda = ConditionalNeuralPotential(
        layers=[LinearLayer(3, 64), ReLU(),
                LinearLayer(64, 32), ReLU(),
                LinearLayer(32, 3)],
        crf_domains=[d_seg_type],
        conditional_dimension=3
    )

    p_d = ConditionalNeuralPotential(
        layers=[LinearLayer(2, 64), ReLU(),
                LinearLayer(64, 32), ReLU(),
                LinearLayer(32, 9)],
        crf_domains=[d_seg_type, d_seg_type],
        conditional_dimension=2,
        conditional_formula=lambda x: np.concatenate((x[:, [0]] - x[:, [1]], x[:, 1:]), axis=1)
    )

    p_dw = MLNPotential(
        formula=lambda x: (np.abs(x[:, 0]) < 0.01) | (x[:, 1] != 0),
        dimension=2,
        w=2
    )

    p_aw = MLNPotential(
        formula=lambda x: (np.abs(x[:, 0]) < 0.5) | (x[:, 1] != 0),
        dimension=2,
        w=2
    )

    p_ao = MLNPotential(
        formula=lambda x: (np.abs(x[:, 0]) < 1.55) | (x[:, 1] == 2),
        dimension=2,
        w=2
    )

    p_dd = MLNPotential(
        formula=lambda x: (x[:, 0] != 1) | (x[:, 1] != 1),
        dimension=2,
        w=2
    )

    p_lw = MLNPotential(
        formula=lambda x: (x[:, 0] > 0.05) | (x[:, 1] != 1),
        dimension=2,
        w=2
    )

    p_prior = ExpWrapper(
        TableFunction(np.log(get_seg_type_distribution(train['seg_type'])))
    )

    f_lda = ParamF(p_lda, atoms=[length('S'), depth('S'), angle('S'), seg_type('S')], lvs=['S'])
    f_d = ParamF(p_d, atoms=[depth('S1'), depth('S2'), seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_neighbor, True))
    # f_dk = ParamF(p_dk, atoms=[length('S1'), depth('S1'), seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_k_aligned))
    # f_dw = ParamF(p_dw, atoms=[depth('S'), seg_type('S')], lvs=['S'])
    f_aw = ParamF(p_aw, atoms=[angle('S'), seg_type('S')], lvs=['S'])
    # f_lw = ParamF(p_lw, atoms=[length('S'), seg_type('S')], lvs=['S'])
    f_ao = ParamF(p_ao, atoms=[angle('S'), seg_type('S')], lvs=['S'])
    f_dd = ParamF(p_dd, atoms=[seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_neighbor, True))
    # f_prior = ParamF(p_prior, atoms=[seg_type('S')], lvs=['S'])

    rel_g = RelationalGraph(
        parametric_factors=[f_lda, f_d, f_aw, f_ao, f_dd]
    )

    g, rvs_dict = rel_g.ground()

    print(len(g.rvs))

    data = dict()
    cond_rvs = set()

    for key, rv in rvs_dict.items():
        if key[0] == length:
            data[rv] = [d_length.clip_value(dt_length[key[1]])]
        if key[0] == angle:
            data[rv] = [dt_angle[key[1]]]
        if key[0] == depth:
            data[rv] = [d_depth.clip_value(dt_depth[key[1]])]
        if key[0] == seg_type:
            data[rv] = [d_seg_type.value_to_idx(dt_seg_type[key[1]])]
        data[rv] = np.array(data[rv])
        if key[0] != seg_type:
            cond_rvs.add(rv)

    g.condition_rvs = cond_rvs

    def visualize(ps, t):
        if t % 200 == 0:
            visualize_2d_potential(p_dd, d_seg_type, d_seg_type, spacing=0.02)

    train_mod(True)
    leaner = PMLE(g, [p_lda, p_d, p_aw, p_ao, p_dd], data)
    leaner.train(
        lr=0.001,
        alpha=0.99,
        regular=0.0001,
        max_iter=3000,
        batch_iter=3,
        batch_size=1,
        rvs_selection_size=200,
        sample_size=30,
        save_dir=f'learned_potentials/model_{model}',
        save_period=1000,
        # visualize=visualize
    )
