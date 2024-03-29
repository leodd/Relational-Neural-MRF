from functions.ExpPotentials import PriorPotential, NeuralNetPotential, \
    CategoricalGaussianFunction, ReLU, LinearLayer, train_mod
from inferer.PBP import PBP
from utils import load
from Graph import *
import numpy as np
from demo.iris_prediction.iris_loader import load_iris_data_fold, matrix_to_dict


train_mod(False)

sepal_length_domain = Domain([4.0, 8.0], continuous=True)
sepal_width_domain = Domain([2.0, 4.5], continuous=True)
petal_length_domain = Domain([1.0, 7.0], continuous=True)
petal_width_domain = Domain([0.1, 2.5], continuous=True)
class_domain = Domain([0, 1, 2], continuous=False)

class_domain.domain_indexize()

rv_sl = RV(sepal_length_domain)
rv_sw = RV(sepal_width_domain)
rv_pl = RV(petal_length_domain)
rv_pw = RV(petal_width_domain)
rv_c = RV(class_domain)

res = list()

for fold in range(5):
    test, _ = load_iris_data_fold('iris', fold, folds=5)
    data = matrix_to_dict(test, rv_sl, rv_sw, rv_pl, rv_pw, rv_c)

    p = CategoricalGaussianFunction(
        domains=[class_domain, sepal_length_domain, sepal_width_domain, petal_length_domain, petal_width_domain],
        index_table=None, weights=None, distributions=None,
        extra_sig=1
    )

    # p = PriorPotential(
    #     NeuralNetPotential(
    #         [LinearLayer(5, 64), ReLU(),
    #          LinearLayer(64, 32), ReLU(),
    #          LinearLayer(32, 1)]
    #     ),
    #     CategoricalGaussianFunction(
    #         domains=[class_domain, sepal_length_domain, sepal_width_domain, petal_length_domain, petal_width_domain],
    #         index_table=None, weights=None, distributions=None,
    #         extra_sig=10
    #     )
    # )

    # p = NeuralNetPotential(
    #     layers=[LinearLayer(5, 64), ReLU(),
    #             LinearLayer(64, 32), ReLU(),
    #             LinearLayer(32, 1)]
    # )

    params = load(
        f'learned_potentials/model_cg/{fold}/3000'
    )

    p.set_parameters(params[0])

    f = F(p, nb=[rv_c, rv_sl, rv_sw, rv_pl, rv_pw])

    g = Graph({rv_sl, rv_sw, rv_pl, rv_pw, rv_c}, {f})

    predict = list()

    M = len(data[rv_c])

    for m in range(M):
        rv_sl.value = data[rv_sl][m]
        rv_sw.value = data[rv_sw][m]
        rv_pl.value = data[rv_pl][m]
        # rv_pw.value = data[rv_pw][m]
        rv_c.value = data[rv_c][m]

        infer = PBP(g, n=20)
        infer.run(2)

        predict.append(infer.map(rv_pw))

    predict = np.array(predict)
    target = data[rv_pw]
    for row in zip(predict, target):
        print(row)
    res.append(np.mean((predict - target) ** 2))
    # res.append(np.mean(predict == target))
    print(res[-1])

print(res, np.mean(res), np.var(res))
