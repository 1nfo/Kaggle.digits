import os
import csv
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
from scipy.optimize import minimize
from scipy import log, exp

_check_grad_ = False
_monitor_ = True
_tuning_ = False

train_valid_sep = 0

cost_hist = []

tuning = (0, 0.1, 5000)  # regular-factor, learning-rate, max-iter
reg_factor = tuning[0]
sample_size = 0
init_range = 1
learning_rate = tuning[1]
input_layer_num = 784
hidden_layer_num = 784
output_layer_num = 10
param_num = (input_layer_num + 1) * hidden_layer_num + (hidden_layer_num + 1) * output_layer_num


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def load(path):
    with open(path, "rb") as f:
        data = []
        csv_r = csv.reader(f)
        count = 0
        for line in csv_r:
            data.append(line)
            count += 1
            if count > sample_size and sample_size:
                break
    return pd.DataFrame(data[1:])


def load_train():
    df = load('data/train.csv')
    yy = df[0]
    xx = df[range(1, input_layer_num + 1)]
    return np.array(xx, dtype="float64"), np.array(yy, dtype="float64")


def load_test():
    return np.array(load('data/test.csv'), dtype="float64")


def param_init(size):
    if os.path.exists("init_param"):
        with open("init_param", "rb") as f:
            result, cost_hist, _, _ = pickle.load(f)
        return result.x
    else:
        res = np.array(np.ones([size, 1]))
        rd.seed(2)
        for ii in res:
            ii *= rd.random()
        return res * init_range - init_range * 0.5


def nn_cost_func(param, xx, yy, reg_factor=0):
    # parameter setting
    m, _ = xx.shape
    theta1 = param[:(input_layer_num + 1) * hidden_layer_num].reshape(input_layer_num + 1,
                                                                      hidden_layer_num).copy()
    theta2 = param[(input_layer_num + 1) * hidden_layer_num:].reshape(hidden_layer_num + 1,
                                                                      output_layer_num).copy()
    # forward
    a1 = np.column_stack((np.ones((m, 1)), np.array(xx, dtype=int)))
    z2 = np.dot(a1, theta1)
    a2 = np.column_stack((np.ones((m, 1)), activated_func(z2)))
    z3 = np.dot(a2, theta2)
    a3 = activated_func(z3)
    y_mat = np.zeros([m, output_layer_num])
    for ii in xrange(len(yy)):
        y_mat[ii, yy[ii]] = 1
    tmp = -y_mat * log(a3) - (1 - y_mat) * log(1 - a3)
    tmp2 = sum(sum(tmp))
    regularization = reg_factor * np.dot(param.T, param) / 2
    j_cost = tmp2 + regularization
    j_cost /= m
    # backward
    delta3 = a3 - y_mat
    delta2 = np.dot(delta3, theta2[1:, :].T) * activated_func(z2) * (1 - activated_func(z2))

    theta1[:, 0] = 0
    theta2[:, 0] = 0
    theta1_grad = (np.dot(a1.T, delta2) + reg_factor * theta1) / m
    theta2_grad = (np.dot(a2.T, delta3) + reg_factor * theta2) / m
    grad = np.hstack([theta1_grad.flatten(), theta2_grad.flatten()])
    return j_cost, grad * learning_rate


def predict(param, xx):
    m, _ = xx.shape
    if m == 0:
        return np.array(None) 
    theta1 = param[:(input_layer_num + 1) * hidden_layer_num].reshape(input_layer_num + 1,
                                                                      hidden_layer_num).copy()
    theta2 = param[(input_layer_num + 1) * hidden_layer_num:].reshape(hidden_layer_num + 1,
                                                                      output_layer_num).copy()
    a1 = np.column_stack((np.ones((m, 1)), np.array(xx, dtype=int)))
    z2 = np.dot(a1, theta1)
    a2 = np.column_stack((np.ones((m, 1)), activated_func(z2)))
    z3 = np.dot(a2, theta2)
    a3 = activated_func(z3)
    return a3.argmax(1)


def activated_func(z):
    res = 1.0 / (1.0 + exp(-z))
    return res


def nn_cost(*args, **kargs):
    return nn_cost_func(*args, **kargs)[0]


def nn_grad(*args, **kargs):
    return nn_cost_func(*args, **kargs)[1]


def monitor(theta):
    if _monitor_:
        cost_hist.append(nn_cost(theta, x, y, reg_factor))
        print (predict(theta, x) == y).mean(), (predict(theta, xv) == yv).mean()


def learning_curve(max_count, step=1):
    sample_count = 1
    trai_errs = []
    vali_errs = []
    for i_step in xrange(len(x)):
        ii = (i_step + 1) * step
        print "new iteration start, sample size:", ii
        rr = minimize(nn_cost, nn_param, args=(x[0:ii], y[0:ii], reg_factor), jac=nn_grad, callback=monitor,
                      method='TNC', options={'maxiter': tuning[2]})
        trai_errs.append((predict(rr.x, x[0:ii]) != y[0:ii]).mean())
        vali_errs.append((predict(rr.x, xv) != yv).mean())
        sample_count += 1
        if sample_count > max_count:
            break
    return trai_errs, vali_errs


if __name__ == "__main__" and not _check_grad_:
    x, y = load_train()
    xv, yv = x[:train_valid_sep, :], y[:train_valid_sep]
    x, y = x[train_valid_sep:, :], y[train_valid_sep:]
    xt = load_test()
    print "Loaded Data"
    nn_param = param_init(param_num)
    if _tuning_:
        a, b = learning_curve(200, 50)
        t = range(1, 1 + len(a))
        plt.plot(t, a)
        plt.plot(t, b)
        plt.show()
    else:
        print "Start Training ..."
        with Timer("test minimizing time"):
            result = minimize(nn_cost, nn_param, args=(x, y, reg_factor), jac=nn_grad, callback=monitor, method='TNC',
                              options={'maxiter': tuning[2]})
            print "Done!"
        print "Training Accuracy:", (predict(result.x, x) == y).mean()
        print "Validation Accuracy:", (predict(result.x, xv) == yv).mean()
        print result
        with open("model", "wb") as f:
            pickle.dump((result, cost_hist, (predict(result.x, x) == y).mean(), (predict(result.x, xv) == yv).mean(), hidden_layer_num), f)

if _check_grad_:
    print "check grad"
    chk_num = 1000
    x, y = load_train()
    nn_param = param_init(param_num)
    new_param = np.array(np.zeros([param_num, 1]))
    grad_h = nn_grad(nn_param, x, y)
    grad_t = []
    count = 0
    for i in new_param:
        count += 1
        if count > chk_num:
            break
        i += 0.00001
        p2 = new_param + nn_param
        i -= 0.00002
        p1 = new_param + nn_param
        grad_t.append((nn_cost(p2, x, y) - nn_cost(p1, x, y)) / 0.00002)
        i += 0.00001
    t = (grad_h[:chk_num] - np.vstack(grad_t).T)
    print sum(sum(t * t))
