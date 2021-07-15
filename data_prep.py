import numpy as np
import cPickle as pickle
import csv


def graph_prep():
    data_file = "costProfCat-LAPOI-top100-withVisitTime.csv"
    dur_fct = 1.0
    walking_speed = 5000.0
    data = {}

    edges = []
    n = 0
    with open(data_file) as f:
        for l in csv.DictReader(f, delimiter=';'):
            edges.append(
                (int(l['from']), int(l['to']), float(l['cost']), float(l['profit']), float(l['visitDuration'])))
            if n < int(l['to']) + 1:
                n = int(l['to']) + 1
    # construct graph
    utility = np.zeros(n)
    duration = np.zeros(n)
    delta = np.empty((n, n))
    delta[:] = np.NaN
    for e in edges:
        delta[e[:2]] = e[2] / walking_speed
        utility[e[1]] = e[3]
        duration[e[1]] = e[4] * dur_fct / 60.0
    vertices = np.where(utility != 0)[0]
    omega_init = set(vertices)
    data['n'] = n
    data['utility'] = utility
    data['duration'] = duration
    data['delta'] = delta
    data['vertices'] = vertices

    with open('graph_la_top100.pkl', 'w') as f:
        pickle.dump((data, omega_init), f)

if __name__ == '__main__':
    graph_prep()
