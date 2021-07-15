import macs_b
import matplotlib.pyplot as plt


class AntColony(macs_b.AntColony):
    acc_bests = []
    count = 0

    dim = 1

    def plot(self, iter_best):
        self.count += 1
        # self.acc_bests.append(abs(iter_best[0]['meal_time'] - self.meal_exp))
        self.acc_bests.append(iter_best[len(iter_best) / 2]['utility'])
        if self.count == self.num_iter:
            self.count = 0
            plt.title('converge curve')
            plt.xlabel('iteration')
            plt.ylabel('utility')
            plt.plot(self.acc_bests)
            plt.show()
