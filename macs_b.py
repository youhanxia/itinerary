import numpy as np
import math
import random
# import time
import cPickle as pickle
import matplotlib.pyplot as plt


class AntColony(object):
    """
    Base line approach
    """

    graph_file = "graph_la_top100.pkl"
    restaurant_file = "restaurant_index_la_top100.pkl"
    num_run = 30
    dur_fct = 1.0

    num_iter = 500
    num_ant = 50
    dim = 3  # utility, meal_time, meal_rating

    start = None
    end = None

    alpha = None
    beta = None
    phi = None
    rho = None
    tau_init = [10000.0, 10.0, 3.0]
    q_0 = 0.1

    # time given in hours
    t_0 = 8.0
    t_1 = 18.0
    meal_exp = 13.0
    meal_dur = 1.0

    is_multi = False

    def __init__(self, se=None):
        self.alpha = 1.0  # 1.0
        self.beta = 0.5  # 0.5
        self.phi = 0.01 / self.num_ant
        self.rho = 0.02

        # read in graph
        with open(self.graph_file) as f:
            # graphData = pickle.load(f)
            self.graph, self.omega_init_0 = pickle.load(f)
        # self.n = self.graph['n']
        # self.utility = self.graph['utility']
        # self.duration = self.graph['duration']
        # self.delta = self.graph['delta']
        # self.vertices = self.graph['vertices']
        
        # read in restaurants
        with open(self.restaurant_file) as f:
            self.restaurant_index = pickle.load(f)

        # initialize tau 
        self.tau = np.empty((self.dim, self.graph['n'], self.graph['n']))
        if self.dim == 1:
            self.ant_weight = [(1.0,) for _ in xrange(self.num_ant)]
        elif self.dim == 2:
            step = math.pi / 2 / (self.num_ant - 1)
            self.ant_weight = [(math.sin(i * step), math.sin((self.num_ant - 1 - i) * step)) for i in xrange(self.num_ant)]
        else:
            self.ant_weight = []
            for _ in xrange(self.num_ant):
                raw_vector = np.random.sample(self.dim)
                unit_vector = raw_vector / np.linalg.norm(raw_vector)
                self.ant_weight.append(list(unit_vector))

        if se:
            self.set_se(se)

    def set_se(self, se):
        self.start = se[0]
        self.end = se[1]
        self.omega_init = self.omega_init_0 - set(se)

    def set_dur(self, dur_fct):
        self.graph['duration'] = np.array([d * dur_fct / self.dur_fct for d in self.graph['duration']])
        self.dur_fct = dur_fct
    
    def run(self):

        # initialize tau in each dimension, initialize before each run if multi-run
        for m in xrange(self.dim):
            self.tau[m][:] = self.tau_init[m]

        # search process
        global_best = []
        for _ in xrange(self.num_iter):
            iter_best = []
            for l in xrange(self.num_ant):
                # generate tour
                meal_time = None
                meal_rating = None
                t = self.t_0
                tour = []
                u = self.start
                omega = set(self.omega_init)
                while True:
                    v = self.next_vertex(omega, l, u, t, meal_time)
                    if v:
                        omega.remove(v)
                        arc = (u, v)
                        tour.append(arc)
                        u = v
                        t += self.graph['delta'][arc]
                        if (not meal_time) and t > self.meal_exp - self.graph['duration'][v] / 2:
                            meal_time = t
                            meal_rating = self.restaurant_index[v][0]['rating']
                            t += self.meal_dur + self.restaurant_index[v][0]['distance'] * 2
                        t += self.graph['duration'][v]
                        if (not meal_time) and t > self.meal_exp:
                            meal_time = t
                            meal_rating = self.restaurant_index[v][0]['rating']
                            t += self.meal_dur + self.restaurant_index[v][0]['distance'] * 2
                    else:
                        break

                arc = (u, self.end)
                tour.append(arc)
                t += self.graph['delta'][arc]
                if (not meal_time) and t > self.meal_exp:
                    meal_time = t
                    meal_rating = self.restaurant_index[u][0]['rating']
                    t += self.meal_dur + self.restaurant_index[u][0]['distance'] * 2

                # iterative improvement
                # self.localSearch(tour, list(omega))

                # local pheromone update
                for m in xrange(self.dim):
                    tau_m = self.tau[m]
                    tau_m[:] *= 1 - self.phi
                    for arc in tour:
                        tau_m[arc] += self.tau_init[m] * self.phi

                # update iteration best
                record = {'tour': tour, 'antID': l, 'timeCost': t - self.t_0, 'meal_time': meal_time, 'meal_rating': meal_rating,
                          'utility': self.get_utility(tour), 'score': self.get_score(tour, meal_time, meal_rating)}
                self.update_best(iter_best, [record])

            # plot according to spec
            if False and not self.is_multi:
                self.plot(iter_best)

            # global pheromone update
            for m in xrange(self.dim):
                dim_best = max(iter_best, key=lambda x: x['score'][m])
                tour = dim_best['tour']
                score = dim_best['score'][m]
                tau_m = self.tau[m]
                tau_m[:] *= 1 - self.rho
                for arc in tour:
                    tau_m[arc] += score * self.rho

            # update global best
            self.update_best(global_best, iter_best)

        return global_best

    def next_vertex(self, omega, l, u, t, meal_time):
        candidates = filter(lambda x: t + self.graph['delta'][u][x] + self.graph['duration'][x] + self.graph['delta'][x][self.end] <= (self.t_1 if meal_time else (self.t_1 - self.meal_dur)), omega)
        if not candidates:
            return None
        fitness = [(v, sum(self.get_fitness(u, v, t, meal_time, m) * self.ant_weight[l][m] for m in xrange(self.dim))) for v in candidates]
        if random.random() < self.q_0:
            return max(fitness, key=lambda x: x[1])[0]
        else:
            dice = random.random() * sum(f for _, f in fitness)
            for v, f in fitness:
                dice -= f
                if dice <= 0.0:
                    return v
        print "error in selecting next nodes"
        exit(1)

    def get_fitness(self, u, v, t, meal_time, m=-1):
        fitness = self.tau[m, u, v] ** self.alpha
        if m == 0:  # utility
            fitness *= (self.graph['utility'][v] / self.graph['delta'][u][v]) ** self.beta
        elif m == 1:  # meal time
            fitness *= 1.0
        elif m == 2:  # meal rating
            fitness *= 1.0
        else:
            fitness *= 1.0
        return fitness

    def get_utility(self, tour):
        return sum(self.graph['utility'][arc[1]] for arc in tour[:-1])

    def get_score(self, tour, meal_time, meal_rating):
        return [self.get_utility(tour), 1.0 / abs(meal_time - self.meal_exp), meal_rating][:self.dim]

    def update_best(self, current_recs, new_recs):
        updated = False
        for new_rec in new_recs:
            comps = [self.dominate(current_rec['score'], new_rec['score']) for current_rec in current_recs]
            for comp, current_rec in zip(comps, current_recs):
                if comp == -1:
                    current_recs.remove(current_rec)
                    updated = True
            if all(comp < 1 for comp in comps):
                current_recs.append(new_rec)
                updated = True
        return updated

    @staticmethod
    def dominate(xs, ys):
        zero = True
        pos = True
        neg = True

        for x, y in zip(xs, ys):
            if x != y:
                zero = False
            if x < y:
                pos = False
            if x > y:
                neg = False

        if zero:
            return 2
        if pos:
            return 1
        if neg:
            return -1
        return 0

    def multi_run(self):
        self.is_multi = True
        result = {'summary': []}
        for i in xrange(self.num_run):
            print '\r', self.__module__, self.start, self.end, self.dur_fct, i,
            result['summary'].append(self.run())
        result['utilities'] = [[rec['utility'] for rec in recs] for recs in result['summary']]
        result['meal_times'] = [[rec['meal_time'] for rec in recs] for recs in result['summary']]
        result['meal_ratings'] = [[rec['meal_rating'] for rec in recs] for recs in result['summary']]

        # with open('results/' + self.__module__ + '_' + (self.start, self.end).__str__() + '_' + self.dur_fct + '_' + str(time.time())[:-3] + '.txt', 'w') as f:
        with open('results/' + self.__module__ + '_' + str((self.start, self.end)) + '_' + str(self.dur_fct) + '.txt', 'w') as f:
            pickle.dump(result, f)
        return result

    def plot(self, iter_best):
        if not iter_best:
            return
        xs = []
        ys = []
        for item in iter_best:
            xs.append(item['utility'])
            ys.append(abs(item['meal_time'] - self.meal_exp))
        plt.hold(True)
        plt.title('pareto front')
        plt.xlabel('utility')
        plt.ylabel('meal time')
        plt.clf()
        plt.scatter(xs, ys)
        plt.xlim([10000, 25000])
        plt.ylim([0.0, 1.0])
        plt.pause(0.0001)
