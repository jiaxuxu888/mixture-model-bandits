import numpy as np
import matplotlib.pyplot as plt
from bandit_class_beta import MixtureBanditBeta
from UCB_known_sd import *
from scipy.stats import beta
from scipy.interpolate import *
from pprint import pprint
from regret_comparison import *


def check_cond(new_para, prev_para, threshold):
    # function that checks whether the algorithm has converged
    # new_para: the new set of parameters
    # prev_para: the previous set of parameters
    # threshold: the threshold

    # OUTPUT:
    # Boolean, TRUE or FALSE

    if new_para == prev_para or np.max([np.abs(new_para), np.abs(prev_para)]) == 0:
        return True
    Kq = np.abs(new_para-prev_para) / np.max([np.abs(new_para), np.abs(prev_para)])
    if Kq < threshold:
        return True
    else:
        return False


def em_beta(x, m, acc, cache_para=None):
    # the EM-M algorithm for solving parameters of a beta mixture
    # x: samples
    # m: components number
    # acc: accuracy for convergence
    # cache_para: previous converged parameters when the algorithm ran last time

    # OUTPUT:
    # a dictionary containing:
    # parameters of a beta mixtures: alpha, beta, weights (pi), mu (component means)
    # var (component variances), and also
    # loop count: how many loops have this algorithm taken to finish?
    N = len(x)
    if cache_para:
        alpha_lst = cache_para['alpha']
        beta_lst = cache_para['beta']
        pi = cache_para['pi']
    else:
        y1 = np.random.uniform(low=0, high=1, size=1)[0]
        y = np.array([y1])
        while len(y) < m:
            candidates = np.array([i for i in x if i not in y])
            Dy = np.array([np.min(np.abs(y - c)) for c in candidates])**2
            new_y_index = list(np.random.multinomial(1, Dy/sum(Dy)))
            new_y_index = new_y_index.index(max(new_y_index))
            y = np.append(y, candidates[new_y_index])
        y.sort()
        alpha_lst = np.random.rand(m)
        beta_lst = np.random.rand(m)
        pi = np.array([1/m]*m)

        for component_num in range(m):
            samples = np.array([sample for sample in x if (sample <= y[component_num]+0.5) and
                                (sample >= y[component_num] - 0.5)])
            # print(samples)
            mu = np.mean(samples)
            if len(samples) <= 1:
                samples = x
            sigma2 = np.std(samples)**2
            phi = mu*(1-mu)/sigma2 - 1
            alpha_lst[component_num] = mu * phi
            beta_lst[component_num] = (1 - mu) * phi

    loop_count = 0
    start_time = time.perf_counter()
    while True:
        loop_count += 1
        W = np.array([[pi[j] * beta.pdf(x[i], a=alpha_lst[j], b=beta_lst[j]) /
                       np.dot(pi, [beta.pdf(x[i], a=alpha_lst[k], b=beta_lst[k])
                                   for k in range(m)])
                       if not np.isnan(pi[j] * beta.pdf(x[i], a=alpha_lst[j], b=beta_lst[j]) /
                       np.dot(pi, [beta.pdf(x[i], a=alpha_lst[k], b=beta_lst[k])
                                   for k in range(m)])) else np.random.uniform(low=0, high=1, size=1)[0]
                       for j in range(m)] for i in range(N)])

        prev_pi, prev_alpha, prev_beta = pi, alpha_lst, beta_lst
        pi = np.sum(W, axis=0) / N
        mu = np.dot(x, W) / (N * pi)
        sigma2 = np.array([np.dot(W[:, j], (x - mu[j])**2) / (N * pi[j]) for j in range(m)])
        phi_lst = mu * (1 - mu) / sigma2 - 1
        alpha_lst, beta_lst = mu * phi_lst, (1 - mu) * phi_lst

        my_cond = True
        for i in range(m):
            if not (check_cond(alpha_lst[i], prev_alpha[i], acc)
                    & check_cond(prev_beta[i], beta_lst[i], acc)
                    & check_cond(prev_pi[i], pi[i], acc)):
                my_cond = False
                break
        if my_cond | (time.perf_counter()-start_time >= 5.0 * 3600 /6/10/1000):
            break

    return {'alpha': alpha_lst, 'beta': beta_lst, 'pi': pi, 'mu': mu, 'var': sigma2,
            'loop_count': loop_count}


def plot_beta_mixture(alphas, betas, pi, label):
    # plot a beta mixture
    x = np.arange(0.01, 0.99, 0.01)
    y = np.array([0.]*len(x))
    for i in range(len(alphas)):
        y += pi[i] * beta.pdf(x, a=alphas[i], b=betas[i])
    plt.plot(x, y, label=label)


def plot_est_and_true(beta_bandit, arm, para, samples):
    # plot a beta mixture given a bandit arm
    # INPUT:
    # beta_bandit: a beta_bandit instance from the MixtureBanditBeta class
    # arm: which arm?
    # para: dictionary containing alphas, betas, and weights (from EM-M algorithm)
    # samples: samples

    # OUTPUT:
    # a plot containing histogram, a theoretical beta mixture densities, an estimated
    # beta density using parameters worked out by EM-M algorithm
    ind = arm - 1
    plot_beta_mixture(alphas=beta_bandit.alpha_list[ind],
                      betas=beta_bandit.beta_list[ind],
                      pi=beta_bandit.pi[ind],
                      label='True Distribution')

    plt.hist(samples, bins=50, density=True)

    plot_beta_mixture(alphas=para['alpha'],
                      betas=para['beta'],
                      pi=para['pi'],
                      label='EM Estimated Distribution')
    plt.legend()
    plt.show()


def ucb_v_beta(bandit, n: int, em=False):
    # UCB-V Algorithm for Beta Mixture Bandits
    # bandit: a beta mixture bandit instance
    # n: horizon
    # em: use EM or not?
    # OUTPUT:
    # [[action 1, reward 1],...,[action n, reward n]]
    k = bandit.arm_num
    m = bandit.components_num
    samples = []
    mu_lst = np.array([0.] * k)
    sd_lst = np.array([0.] * k)
    cache_alphas = np.random.rand(k, m)
    cache_betas = np.random.rand(k, m)
    cache_pi = np.random.rand(k, m)
    count_lst = [0] * k
    loop_count = ['init']

    each_arm_sample_size = int(np.min([m * 10, np.floor(n / k)]))
    for i in range(k):
        initial_samples = bandit.pull(arm=i+1, size=each_arm_sample_size)
        for sample in initial_samples:
            samples.append([i+1, sample])
        mu_lst[i] = np.mean(initial_samples)
        sd_lst[i] = np.std(initial_samples)
        count_lst[i] += each_arm_sample_size

    ucb_lst = 0
    for t in range(k*each_arm_sample_size, n):
        if em:
            if t % 10 == 0:
                print(f"EM-beta, t = {t}, bernstein bounds = {ucb_lst}, count = {count_lst},"
                      f" loop count is {loop_count[-1]}")
        else:
            if t % 200 == 0:
                print(f"Non-EM-beta, t = {t}, bernstein bounds = {ucb_lst}, count = {count_lst}")

        ucb_lst = [ucb_bern_bound(t=t, big_t=count_lst[i],
                                  mu=mu_lst[i],
                                  var=sd_lst[i] ** 2,
                                  b=1) for i in range(k)]
        action = ucb_lst.index(max(ucb_lst)) + 1
        new_x = bandit.pull(arm=action, size=1)[0]
        samples.append([action, new_x])
        count_lst[action-1] += 1
        prev_samples = np.array([x[1] for x in samples if x[0] == action])
        mu_lst[action-1] = np.mean(prev_samples)
        if not em:
            sd_lst[action-1] = np.std(prev_samples)
        else:
            if len(prev_samples) <= 50:
                para = em_beta(prev_samples, m, acc=0.1)
            else:
                para = em_beta(prev_samples, m, acc=0.1,
                               cache_para={'alpha': cache_alphas[action-1],
                                           'beta': cache_betas[action-1],
                                           'pi': cache_pi[action-1]})
            var = np.dot(para['pi'], para['var']+para['mu']**2) - mu_lst[action-1]**2
            sd_lst[action - 1] = np.sqrt(var)
            cache_alphas[action-1] = para['alpha']
            cache_betas[action-1] = para['beta']
            cache_pi[action-1] = para['pi']
            loop_count.append(para['loop_count'])
    print(count_lst)
    return samples


def ucb_v_em_beta(bandit, n):
    return ucb_v_beta(bandit, n, em=True)


# sample = my_beta_mixture.pull(arm=1, size=5000)
# print(my_beta_mixture.parameters(latex=True))
#
# para_est = em_beta(sample, m=my_beta_mixture.components_num, acc=0.01)
# plot_est_and_true(my_beta_mixture, arm=1, para=para_est, samples=sample)


def make_plots_beta(k_lst, components_lst, n_lst, alpha_range_lst, beta_range_lst,
                    rep, folder_name, algorithm_lst, plots_num=1):
    # function for drawing and saving multiple plots and parameters
    # k_lst: a list containing the number of arms an environment carries
    # components_lst: a list containing possible numbers of components
    # n_lst: a list containing horizons
    # alpha_range_list: list of the form [[a1, b1], ..., [an, bn]] denoting the range of alphas
    # beta_range_list: list of the form [[a1, b1], ..., [an, bn]] denoting the range of betas
    # rep: how many experiments to repeat under a given environment
    # folder_name: folder name
    # algorithm_lst: which algorithms to test on? of the form [[algorithm, name],...]
    # plots_num: repeat all for how many plots?
    for a_plot in range(plots_num):
        for k in k_lst:
            for index in range(len(alpha_range_lst)):
                for m in components_lst:
                    print(f"k = {k}, m = {m}, alpha range is {alpha_range_lst[index]}, "
                          f"beta range is {beta_range_lst[index]}".center(130, '@'))
                    my_beta_mixture = MixtureBanditBeta(arm_num=k,
                                                        alpha_range=alpha_range_lst[index],
                                                        beta_range=beta_range_lst[index],
                                                        components_num=m)
                    regret_comparison(bandit=my_beta_mixture,
                                      algorithm_list=algorithm_lst,
                                      n=n_lst[index], rep=rep, show=False, folder_name=folder_name)


make_plots_beta(k_lst=[2],
                alpha_range_lst=[[0, 2], [0, 10]],
                beta_range_lst=[[0, 2], [0, 10]],
                n_lst=[1000, 1000],
                components_lst=[3],
                rep=10,
                folder_name='Beta_mixture_m_3',
                algorithm_lst=[[ucb_v_beta, 'ucb-v-beta'],
                               [ucb_v_em_beta, 'ucb-v-em-beta']])
