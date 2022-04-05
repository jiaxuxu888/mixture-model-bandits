import numpy as np
from scipy.stats import norm, uniform
from sklearn.cluster import KMeans
from bandit_class import MixtureBandit


def ucb_bound(t, prev_t, mu, sd, alpha_value):
    # This is the upper confidence bound for unknown variance case
    # input: parameters for the bound
    # output: the UCB bound
    def ft(x): return 1 + x * np.log(x) ** 2
    if alpha_value is not None:
        var_est = np.dot(alpha_value, sd ** 2)
        return mu + np.sqrt(2 * var_est * np.log(ft(t)) / prev_t)
    return mu + np.sqrt(2 * np.max(sd ** 2) * np.log(ft(t)) / prev_t)


def em_known_var(x, sd, m, acc, prev_mean, prev_alpha):
    # EM algorithm for known variance case
    # INPUT:
    # x: samples,
    # sd: standard deviations,
    # m: components number
    # acc: accuracy criterion for convergence
    # prev_mean: component means when the algorithm converged last time
    # prev_alpha: weights when the algorithm converged last time

    # OUTPUT:
    # [component means, component weights]
    x = np.array(x)
    big_n = len(x)
    if (np.min(prev_alpha) > 0.01) & (np.min(1 - prev_alpha) > 0.01) & \
            (np.min(np.abs(prev_alpha - 1/m)) > 0.01) & (big_n > 50):
        mean = prev_mean
        alpha = prev_alpha
    else:
        k_means = KMeans(n_clusters=m, random_state=0).fit(x.reshape(-1, 1))
        mean = np.squeeze(k_means.cluster_centers_)
        alpha = uniform.rvs(loc=0.4, scale=0.3, size=m)

    while True:
        p = np.array([norm.pdf(x, loc=mean[i], scale=sd[i]) for i in range(m)])
        p_x = np.array([np.dot([p[j][i] for j in range(m)], alpha) for i in range(big_n)])
        w = np.array([alpha[i] * p[i] / p_x for i in range(m)])

        new_alpha = np.array([np.sum(w[i]) for i in range(m)]) / big_n
        new_mean = np.array([1 / np.sum(w[i]) * np.dot(w[i], x) for i in range(m)])

        if (np.max(np.abs(mean - new_mean)) < acc) & (np.max(np.abs(new_alpha - alpha)) < acc):
            break
        alpha = new_alpha
        mean = new_mean
    return [mean, alpha]


def ucb(bandit: MixtureBandit, n: int):
    # UCB algorithm for the known variance case
    # INPUT:
    # bandit: a bandit instance from MixtureBandit class
    # n: horizon

    # OUTPUT:
    # [[action 1, reward 1],...,[action n, reward n]]
    k = bandit.arm_num  # how many arms?
    m = bandit.components_num
    samples = []
    mu_lst = np.array([0.] * k)  # means of each arm
    sd = np.array(bandit.sd_list)
    count_lst = [0] * k

    size = np.max([m, 10])
    for i in range(k):
        sample = bandit.pull(arm=i + 1, size=size)
        for j in range(size):
            samples.append([i + 1, sample[j]])
        count_lst[i] += size
        mu_lst[i] = np.mean(sample)

    for t in range(k * size, n):
        if t % 100 == 0:
            print('t = ', t)
        ucb_lst = [ucb_bound(t=t,
                             prev_t=count_lst[j],
                             mu=mu_lst[j],
                             sd=sd[j],
                             alpha_value=None) for j in range(k)]

        action = ucb_lst.index(max(ucb_lst)) + 1
        count_lst[action - 1] += 1
        new_x = bandit.pull(arm=action, size=1)
        samples.append([action, new_x])
        mu_lst[action - 1] = np.mean([x[1] for x in samples if x[0] == action])
    print(count_lst)
    return samples


def ucb_em(bandit: MixtureBandit, n: int):
    # UCB-EM Algorithm for the known variance case
    # INPUT
    # bandit: a bandit instance from the MixtureBandit class
    # n: horizon

    # OUTPUT
    # [[action 1, reward 1],...,[action n, reward n]]
    print(bandit.parameters(latex=False))
    k = bandit.arm_num
    m = bandit.components_num
    samples = []
    mu_lst = np.array([0.] * k)
    sd = bandit.sd_list

    prev_means = np.array([[0.] * m] * k)
    prev_alphas = np.random.dirichlet([1]*m, size=k)
    count_lst = [0] * k

    size = np.max([10, m])
    for i in range(k):
        sample = bandit.pull(arm=i + 1, size=size)
        count_lst[i] += size
        for j in range(size):
            samples.append([i + 1, sample[j]])
        mu_lst[i] = np.mean(sample)
        prev_means[i] = [np.mean(sample)]*m

    action = 0
    ucb_lst_em = 0
    for t in range(k * size, n):  # do the UCB algorithm for remaining of t from k+1 to n
        if t % 100 == 0:
            print(f"t = {t}, chosen arm: {action}, ucb_em: {ucb_lst_em} count_lst: {count_lst}")

        ucb_lst_em = [ucb_bound(t=t, prev_t=count_lst[j], mu=mu_lst[j], sd=sd[j],
                                alpha_value=prev_alphas[j]) for j in range(k)]

        action = ucb_lst_em.index(max(ucb_lst_em)) + 1
        new_x = bandit.pull(arm=action, size=1)[0]
        samples.append([action, new_x])
        count_lst[action-1] += 1

        prev_samples = [x[1] for x in samples if x[0] == action]
        [means, alpha] = em_known_var(x=prev_samples,
                                      sd=sd[action - 1],
                                      m=m,
                                      acc=1e-3,
                                      prev_mean=prev_means[action - 1],
                                      prev_alpha=prev_alphas[action - 1])
        prev_means[action - 1] = means
        prev_alphas[action - 1] = alpha
        mu_lst[action - 1] = np.mean([x[1] for x in samples if x[0] == action])
    print(count_lst)
    return samples
