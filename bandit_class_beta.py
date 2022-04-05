import numpy as np
from scipy.stats import norm, uniform
from tabulate import tabulate
from pprint import pprint


class MixtureBanditBeta:
    # the Beta mixture model Bandit class, the structure is
    # mostly the same as 'MixtureBandit'
    def __init__(self, arm_num, alpha_range, beta_range, components_num=2):
        self.is_beta = True

        self.arm_num = arm_num
        self.components_num = components_num

        self.alpha_range = alpha_range
        self.beta_range = beta_range

        self.pi = np.random.dirichlet([1]*components_num, size=arm_num)
        self.alpha_list = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=[arm_num, components_num])
        self.beta_list = np.random.uniform(low=beta_range[0], high=beta_range[1], size=[arm_num, components_num])

        self.mean_list = self.alpha_list/(self.alpha_list+self.beta_list)
        self.var_list = self.alpha_list*self.beta_list/\
                        (self.alpha_list+self.beta_list)**2\
                        /(self.alpha_list+self.beta_list+1)

        self.weighted_mean = [np.dot(self.mean_list[i], self.pi[i]) for i in range(self.arm_num)]
        self.optimal_arm_mean = max(self.weighted_mean)
        self.optimal_arm = self.weighted_mean.index(self.optimal_arm_mean) + 1

    def print_optimal_arm(self):
        print(f"\nBest arm is the arm {self.optimal_arm} "
              f"with mean {self.optimal_arm_mean:1.3f}\n")

    def get_para(self):
        return {'alpha': self.alpha_list, 'beta': self.beta_list, 'pi': self.pi}

    def pull(self, arm, size):
        arm_alpha = self.alpha_list[arm-1]
        arm_beta = self.beta_list[arm-1]
        my_sample = []
        for i in range(size):
            component_chosen = list(np.random.multinomial(1, self.pi[arm-1]))
            component_chosen = component_chosen.index(max(component_chosen))
            my_sample.append(np.random.beta(a=arm_alpha[component_chosen], b=arm_beta[component_chosen], size=size)[0])
        return my_sample

    def parameters(self, latex: bool):
        table = list([])

        for component_index in range(self.components_num):
            alpha = ['alpha '+str(component_index+1)] + \
                    list(np.round([self.alpha_list[i][component_index]
                                   for i in range(self.arm_num)], 2))
            beta = ['beta ' + str(component_index + 1)] + \
                   list(np.round([self.beta_list[i][component_index]
                                  for i in range(self.arm_num)], 2))
            pi = ['pi ' + str(component_index + 1)] + \
                 list(np.round([self.pi[i][component_index]
                                for i in range(self.arm_num)], 2))
            table.append(alpha)
            table.append(beta)
            table.append(pi)
            table.append(['-']+['']*self.arm_num)
        table.append(['weighted mean'] + list(np.round(self.weighted_mean, 2)))
        headers = ['arm ' + str(x + 1) if x + 1 != self.optimal_arm
                   else 'arm ' + str(x + 1) + '*' for x in range(self.arm_num)]
        if latex:
            return tabulate(table, headers=headers, tablefmt='latex')
        return tabulate(table, headers=headers, tablefmt='simple')


# my_bandit = MixtureBanditBeta(arm_num=3, components_num=2, alpha_range=[0, 5], beta_range=[0, 5])
# print(my_bandit.parameters(latex=False))
# my_bandit.print_optimal_arm()
# print(my_bandit.pull(arm=1, size=10))
#

