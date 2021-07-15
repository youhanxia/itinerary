import macs_b

class AntColony(macs_b.AntColony):
    dim = 2
    normalizer = 1000 / 5.0
    bias = -2.5

    def get_score(self, tour, meal_time, meal_rating):
        return [self.get_utility(tour) + self.normalizer * (meal_rating + self.bias), 1.0 / abs(meal_time - self.meal_exp), meal_rating][:self.dim]
