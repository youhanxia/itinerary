import cPickle as pickle
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, ttest_rel


def star(p):
    if p <= 0.0001:
        return '****'
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    return 'ns'


def main0():
    sys.stderr = open('nul', 'w')

    with open("tuning_res_b2.pkl", 'r') as f:
        res = pickle.load(f)

    bln = res['alpha'][2][1]

    for item in ['alpha', 'beta', 'phi', 'rho']:
        print '$\\' + item + '$',

        for a in res[item]:
            print '&', a[0],
        print '\\\\'

        for a in res[item]:
            a_mean = np.mean(a[1])
            print '&', '%.2f' % a_mean,
        print '\\\\'

        for a in res[item]:
            print '&', star(ttest_rel(a[1], bln, nan_policy='omit')[1]),
        print '\\\\'

        print '\\hline'

    plt.show()


if __name__ == '__main__':
    main0()
