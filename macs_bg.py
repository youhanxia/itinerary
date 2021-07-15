import acs_g


class AntColony(acs_g.AntColony):
    dim = 2

    def get_score(self, tour, meal_time, meal_rating):
        return [self.get_utility(tour), (self.get_utility(tour) + self.normalizer * (meal_rating + self.bias)) *
                self.penalty(meal_time, self.meal_exp)][:self.dim]

    def get_fitness(self, u, v, t, meal_time, m=-1):
        fitness = self.tau[m, u, v] ** self.alpha
        if m == 0 or m == 1:  # utility
            fitness *= (self.graph['utility'][v] / self.graph['delta'][u][v]) ** self.beta
        else:
            fitness *= 1.0
        return fitness
