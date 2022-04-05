import numpy as np
from scipy.stats import norm, uniform
from tabulate import tabulate
from pprint import pprint


class MixtureBandit:
    # Mixture Model Bandit class
    def __init__(self, arm_num, components_num=2, mean_range=None, sd_range=None):
        # arm_num: k, how many arms
        # components_num: how many components
        # mean_range: the range from which the component means are sampled from
        # sd_range: the range from which the component standard deviations are sampled from

        self.is_beta = False

        self.arm_num = arm_num
        self.components_num = components_num

        self.mean_range = mean_range
        self.sd_range = sd_range

        self.alpha = np.random.dirichlet([1]*components_num, size=arm_num)
        self.mean_list = np.random.uniform(low=mean_range[0], high=mean_range[1],size=[arm_num, components_num])
        self.sd_list = np.random.uniform(low=sd_range[0], high=sd_range[1],size=[arm_num, components_num])

        self.weighted_mean = [np.dot(self.alpha[i], self.mean_list[i]) for i in range(self.arm_num)]
        self.optimal_arm_mean = max(self.weighted_mean)
        self.optimal_arm = self.weighted_mean.index(self.optimal_arm_mean) + 1

    def print_optimal_arm(self):
        print(f"\nBest arm is the arm {self.optimal_arm} "
              f"with mean {self.optimal_arm_mean:1.3f}\n")

    def get_para(self):
        # output the parameters of the environment
        return {'mean': self.mean_list, 'sd': self.sd_list, 'alpha': self.alpha}

    def pull(self, arm, size):
        # get rewards from an arm
        arm_mean = self.mean_list[arm-1]
        arm_sd = self.sd_list[arm-1]
        my_sample = []
        for i in range(size):
            component_chosen = list(np.random.multinomial(1, self.alpha[arm-1]))
            component_chosen = component_chosen.index(max(component_chosen))
            my_sample.append(norm.rvs(loc=arm_mean[component_chosen], scale=arm_sd[component_chosen],size=size)[0])
        return my_sample

    def parameters(self, latex: bool):
        # return a tabulate that displays the parameters of a bandit environment
        table = list([])

        for component_index in range(self.components_num):
            mean = ['mean '+str(component_index+1)] + list(np.round([self.mean_list[i][component_index] for i in range(self.arm_num)], 2))
            sd = ['std ' + str(component_index + 1)] + list(np.round([self.sd_list[i][component_index] for i in range(self.arm_num)], 2))
            alpha = ['alpha ' + str(component_index + 1)] + list(np.round([self.alpha[i][component_index] for i in range(self.arm_num)], 2))
            table.append(alpha)
            table.append(mean)
            table.append(sd)
            table.append(['-']+['']*self.arm_num)
        table.append(['weighted mean'] + list(np.round(self.weighted_mean, 2)))
        headers = ['arm ' + str(x + 1) if x + 1 != self.optimal_arm
                   else 'arm ' + str(x + 1) + '*' for x in range(self.arm_num)]
        if latex:
            return tabulate(table, headers=headers, tablefmt='latex')
        return tabulate(table, headers=headers, tablefmt='simple')


# my_bandit = MixtureBandit(arm_num=2, components_num=3, mean_range=[0, 10], sd_range=[0.5, 1])
# print(my_bandit.parameters(latex=True))
# my_bandit.print_optimal_arm()
# print(np.mean(my_bandit.pull(arm=0, size=1000)))
