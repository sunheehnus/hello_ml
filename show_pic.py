#!/usr/bin/env python
# encoding: utf-8

def showLogRegres(train_data_path, theta_data_path):
    import matplotlib.pyplot as plt

    lines = map(lambda x: map(lambda y: float(y), x), map(lambda x: x.split("\t"), open(train_data_path).readlines()))

    for params in lines:
        if int(params[2]) == 0:
            plt.plot(params[0], params[1], 'or')
        elif int(params[2]) == 1:
            plt.plot(params[0], params[1], 'ob')
        params = map(lambda x: float(x), params)

    min_x = min(map(lambda x: x[0], lines))
    max_x = max(map(lambda x: x[0], lines))
    thetas = map(lambda x: float(x), open(theta_data_path).readline().split("\t")[0:3])
    y_min_x = float(-thetas[0] - thetas[1] * min_x) / thetas[2]
    y_max_x = float(-thetas[0] - thetas[1] * max_x) / thetas[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.show()
showLogRegres("train_data", "theta_data")
