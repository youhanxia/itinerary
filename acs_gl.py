import acs_g
import random


class AntColony(acs_g.AntColony):
    dim = 1

    def __init__(self, se=None):
        super(AntColony, self).__init__(se)
        self.meal_ignore = self.meal_exp - 0.5

    def next_vertex(self, omega, l, u, t, meal_time):
        candidates = filter(lambda x: t + self.graph['delta'][u][x] + self.graph['duration'][x] + self.graph['delta'][x][self.end] <= (self.t_1 if meal_time else (self.t_1 - self.meal_dur)), omega)
        if not candidates:
            return None
        times = [(v, t + self.graph['delta'][u][v]) for v in candidates]
        if min(times, key=lambda x: x[1])[1] < self.meal_ignore:  # too early to consider meal
            candidates = [v for v, t in times if t < self.meal_ignore]
            fitness = [(v, self.get_fitness(u, v, t, meal_time, 0)) for v in candidates]
        elif meal_time:  # already had meal
            fitness = [(v, self.get_fitness(u, v, t, meal_time, 0)) for v in candidates]
        else:  # to consider meal
            fitness = [(v, self.get_fitness(u, v, t, meal_time, 1)) for v in candidates]

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
        fitness = self.tau[0, u, v] ** self.alpha
        if m == 0:  # utility
            fitness *= (self.graph['utility'][v] / self.graph['delta'][u][v]) ** self.beta
        elif m == 1:  # to have meal
            fitness *= ((self.restaurant_index[v][0]['rating'] + self.bias) *
                        (self.penalty(t + self.graph['delta'][u][v], self.meal_exp) +
                         self.penalty(t + self.graph['delta'][u][v] + self.graph['duration'][v], self.meal_exp))) ** self.beta
        else:
            fitness *= 1.0
        return fitness
