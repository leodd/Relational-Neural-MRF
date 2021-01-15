from functions.ExpPotentials import CGNeuralNetPotential, ReLU, LinearLayer
from learner.NeuralPMLEHybrid import PMLE
from Graph import *
from demo.iris_prediction.iris_loader import load_iris_data_fold, matrix_to_dict


sepal_length_domain = Domain([4.0, 8.0], continuous=True)
sepal_width_domain = Domain([2.0, 4.5], continuous=True)
petal_length_domain = Domain([1.0, 7.0], continuous=True)
petal_width_domain = Domain([0.1, 2.5], continuous=True)
class_domain = Domain([0, 1, 2], continuous=False)

rv_sl = RV(sepal_length_domain)
rv_sw = RV(sepal_width_domain)
rv_pl = RV(petal_length_domain)
rv_pw = RV(petal_width_domain)
rv_c = RV(class_domain)

for fold in range(5):
    train, _ = load_iris_data_fold('iris', fold, folds=5)
    data = matrix_to_dict(train, rv_sl, rv_sw, rv_pl, rv_pw, rv_c)

    p = CGNeuralNetPotential(
        layers=[LinearLayer(5, 64), ReLU(),
                LinearLayer(64, 32), ReLU(),
                LinearLayer(32, 1)],
        domains=[class_domain, sepal_length_domain, sepal_width_domain, petal_length_domain, petal_width_domain],
        extra_sig=10
    )

    # p = NeuralNetPotential(
    #     layers=[LinearLayer(5, 64), ReLU(),
    #             LinearLayer(64, 32), ReLU(),
    #             LinearLayer(32, 1)]
    # )

    f = F(p, nb=[rv_c, rv_sl, rv_sw, rv_pl, rv_pw])

    g = Graph({rv_sl, rv_sw, rv_pl, rv_pw, rv_c}, {f})

    def visualize(ps, t):
        if t % 200 == 0:
            pass

    leaner = PMLE(g, [p], data)
    leaner.train(
        lr=0.0001,
        alpha=0.999,
        regular=0.0001,
        max_iter=3000,
        batch_iter=3,
        batch_size=50,
        rvs_selection_size=5,
        sample_size=30,
        # visualize=visualize,
        save_dir=f'learned_potentials/model_2/{fold}',
        save_period=1000,
    )
