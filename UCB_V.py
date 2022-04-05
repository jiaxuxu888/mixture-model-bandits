import numpy as np
from bandit_class import MixtureBandit
from sklearn.mixture import GaussianMixture
from UCB_known_sd import *


def gm_sol(x, m):
    # Function for solving gaussian mixture parameters
    # INPUT:
    # x: samples from an arm
    # m: how many components?
    # OUTPUT:
    # sd: estimated overall standard deviation of the arm
    x = np.array(x)
    x_bar = np.mean(x)
    gm = GaussianMixture(n_components=m, random_state=0, warm_start=True).fit(x.reshape(-1, 1))
    component_means = gm.means_.reshape(m)
    component_vars = gm.covariances_.reshape(m)
    my_cov = np.dot(component_means**2+component_vars, gm.weights_) - x_bar**2
    return {'sd': np.sqrt(my_cov)}


def ucb_bern_bound(mu, var, b, t, big_t):
    # UCB bernstein bound
    return mu + np.sqrt(2 * var * np.log(t) / big_t) + b * np.log(t) / (2*big_t)


def ucb_v(bandit: MixtureBandit, n: int, em=False):
    # UCB-V Algorithm
    # bandit: a bandit instance from the MixtureBandit class
    # n: horizon
    # em: use EM algorithm or not?
    # OUTPUT:
    # [[action 1, reward 1],...,[action n, reward n]]
    k = bandit.arm_num
    m = bandit.components_num
    samples = []
    mu_lst = [0] * k
    sd_lst = [0] * k
    count_lst = [0] * k
    b = 1

    for i in range(k):
        initial_samples = bandit.pull(arm=i+1, size=m-1)
        if max(initial_samples) > b:
            b = max(initial_samples)
        for sample in initial_samples:
            samples.append([i+1, sample])
        mu_lst[i] = np.mean(initial_samples)
        sd_lst[i] = 1
        count_lst[i] += m-1

    ucb_lst = 0
    for t in range(k*(m-1), n):
        if t % 200 == 0:
            print(f"t = {t}, bernstein bounds = {ucb_lst}, count = {count_lst}")

        ucb_lst = [ucb_bern_bound(t=t, big_t=count_lst[i],
                                  mu=mu_lst[i],
                                  var=sd_lst[i] ** 2,
                                  b=b) for i in range(k)]
        action = ucb_lst.index(max(ucb_lst)) + 1
        new_x = bandit.pull(arm=action, size=1)[0]
        if new_x > b:
            b = new_x
        samples.append([action, new_x])
        count_lst[action-1] += 1

        prev_samples = [x[1] for x in samples if x[0] == action]
        mu_lst[action-1] = np.mean(prev_samples)
        if not em:
            sd_lst[action-1] = np.std(prev_samples)
        else:
            solution = gm_sol(prev_samples, m)
            sd_lst[action-1] = solution['sd']
    print(count_lst, ', b = ', b)
    return samples


def ucb_v_em(bandit, n):
    return ucb_v(bandit, n, em=True)
