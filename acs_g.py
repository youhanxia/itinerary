import acs_b
import math


class AntColony(acs_b.AntColony):
    dim = 1

    bias = -2.5
    normalizer = 1000 / 5.0
    width = 0.25

    penaltyMode = 'Normal'  # normal, 1st_order, square

    def penalty(self, x, mu=0, width=None):
        if not width:
            width = self.width
        if self.penaltyMode == 'Normal':
            return math.exp(-(((x - mu) / width) ** 2))
        elif self.penaltyMode == '1st_order':
            return math.exp(-abs(x - mu) / width)
        elif self.penaltyMode == 'square':
            return 1.0 if abs(x - mu) < width else 0.1
        else:
            return 1.0

    def get_score(self, tour, meal_time, meal_rating):
        return [(self.get_utility(tour) + self.normalizer * (meal_rating + self.bias)) *
                self.penalty(meal_time, self.meal_exp), 1.0 / abs(meal_time - self.meal_exp), meal_rating][:self.dim]
