import acs_b
import acs_g
import acs_gl
import macs_b
import macs_bg
import macs_gt
import random
from matplotlib import pyplot as plt
import cPickle as pickle
# from scipy.stats import ttest_ind, ttest_rel
# import numpy as np


ses = [(155, 154), (232, 124), (141, 387), (55, 157), (378, 137), (337, 273), (295, 188), (39, 272), (219, 55),
       (264, 182), (271, 328), (73, 102), (150, 389), (15, 76), (339, 390), (202, 391), (41, 304), (414, 26),
       (294, 220), (157, 31), (321, 167), (390, 141), (286, 394), (260, 388), (76, 293), (99, 39), (137, 321),
       (394, 260), (10, 155), (309, 150)]
dur_factors = [0.5, 1.0, 2.0]
models = {}


def run_for_results():
    global ses, dur_factors, models

    for m in models.itervalues():
        for se in ses:
            for dur_fct in dur_factors:
                m.set_se(se)
                m.set_dur(dur_fct)
                m.multi_run()


def test_on_strategy():
    global ses, dur_factors, models
    i = 0
    while True:
        se = random.sample(ses, 1)[0]
        model_m = models['m']
        model_s = models['g']
        model_m.set_se(se)
        model_s.set_se(se)
        solutions_m = model_m.run()
        solution_s = model_s.run()[0]
        utilities = [solution['utility'] for solution in solutions_m]
        meal_time_errors = [abs(13.0 - solution['meal_time']) for solution in solutions_m]
        plt.figure(figsize=(15, 15))
        plt.plot(utilities, meal_time_errors, '.', label='M-ItiMeal')
        plt.plot(solution_s['utility'], abs(13.0 - solution_s['meal_time']), '*', label='S-ItiMeal')
        plt.legend(loc='best')
        plt.xlabel('utility')
        plt.ylabel('meal time error (h)')
        plt.gca().invert_yaxis()
        plt.savefig('imgs/front_vs_single_' + str(se) + '_' + str(i))
        i += 1


def tune_parameters():
    global ses, dur_factors, models
    model = models['b']

    with open("tuning_res_b2.pkl", 'r') as f:
        res = pickle.load(f)
    # res = dict()

    values = dict()
    values['alpha'] = [10.0] # [0.1, 0.3, 1.0, 3.0]
    values['beta'] = [10.0] # [0.1, 0.3, 1.0, 3.0]
    values['phi'] = [] # [0.0003, 0.001, 0.003, 0.01, 0.03]
    values['rho'] = [] # [0.1, 0.001, 0.003, 0.01, 0.03]

    for item in ['alpha', 'beta', 'phi', 'rho']:
        # res[item] = []
        model.__init__()
        for val in values[item]:
            if item == 'phi':
                setattr(model, item, val / model.num_ant)
            else:
                setattr(model, item, val)
            temp_res = []
            for se in ses:
                model.set_se(se)
                print item, val, se,
                temp_res.append(model.run()[0]['score'][0])
                print '\r',
            res[item].append((val, temp_res))

    with open("tuning_res_b2.pkl", 'w') as f:
        pickle.dump(res, f)


def main():
    global ses, dur_factors, models
    models['b'] = acs_b.AntColony()
    models['g'] = acs_g.AntColony()
    models['gl'] = acs_gl.AntColony()
    models['m'] = macs_b.AntColony()
    models['mg'] = macs_bg.AntColony()
    models['mt'] = macs_gt.AntColony()

    # run_for_results()
    # test_on_strategy()
    tune_parameters()


if __name__ == '__main__':
    main()
