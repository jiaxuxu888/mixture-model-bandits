import numpy as np
from UCB_known_sd import *
from UCB_V import gm_sol


def ucb_normal(bandit: MixtureBandit, n: int, em=False):
    # UCB-NORMAL algorithm
    # INPUT:
    # bandit: a bandit instance from the MixtureBandit class
    # n: horizon
    # em: boolean, use EM or not?
    # OUTPUT:
    # [[action 1, reward 1],...,[action n, reward n]]
    k = bandit.arm_num
    samples = []
    count_lst = [0] * k
    sd_lst = [1] * k
    bounds = 0
    for t in range(1, n+1):
        if t % 200 == 0:
            print(f"t = {t}, bounds = {bounds}, count = {count_lst}")
        check_log = count_lst < np.ceil(8 * np.log(t))
        if t == 1:
            check_log[0] = True
        cond = False
        for i in range(1, k+1):
            if check_log[i-1]:
                samples.append([i, bandit.pull(arm=i, size=1)[0]])
                count_lst[i-1] += 1
                cond = True
                break
        if cond:
            continue

        bounds = [0]*k
        for arm in range(1, k+1):
            prev_samples = np.array([x[1] for x in samples if x[0] == arm])
            n = len(prev_samples)
            x_mean = np.mean(prev_samples)

            if em:
                bounds[arm - 1] = x_mean + np.sqrt(16 * np.square(sd_lst[arm-1]) * np.log(t - 1) / n)
            else:
                q_j = np.sum(np.square(prev_samples))
                bounds[arm - 1] = x_mean + np.sqrt(16 * q_j/(n - 1) * np.log(t-1)/n)

        action = bounds.index(max(bounds)) + 1
        new_x = bandit.pull(arm=action, size=1)[0]
        if em:
            prev_samples = np.array([x[1] for x in samples if x[0] == action])
            solutions = gm_sol(prev_samples, bandit.components_num)
            sd_lst[action-1] = solutions['sd']
        samples.append([action, new_x])
        count_lst[action - 1] += 1
    print(count_lst)
    return samples


def ucb_normal_em(bandit, n):
    return ucb_normal(bandit, n, em=True)

