from sklearn.mixture import BayesianGaussianMixture
from bandit_class import MixtureBandit
from scipy.stats import norm
from pprint import pprint
import numpy as np


def posterior_para(sample, bandit):
    # Posterior of the parameters
    # INPUT:
    # sample: samples
    # bandit: a bandit instance from the bandit class
    # OUTPUT:
    # a dictionary containing:
    # 'dirichlet_concentration': concentration parameters of the posterior
    #                            dirichlet distribution for the weights
    # 'mean': means of the posterior of component mean
    # 'mean_sd': standard deviations of the posterior of component mean
    x = np.squeeze(np.array(sample)).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=bandit.components_num,
                                  mean_precision_prior=0.1,
                                  covariance_prior=[[np.mean(bandit.sd_range)**2]],
                                  max_iter=5000,
                                  weight_concentration_prior_type='dirichlet_distribution',
                                  warm_start=True).fit(x)

    return {'dirichlet_concentration': np.squeeze(bgm.weight_concentration_),
            'mean': np.squeeze(bgm.means_),
            'mean_sd': 1/np.sqrt(bgm.mean_precision_)}


def thompson_sampling(bandit: MixtureBandit, n: int):
    # Thompson sampling algorithm
    # INPUT:
    # bandit: a bandit instance from the bandit class
    # n: horizon
    # OUTPUT:
    # [[action 1, reward 1],...,[action n, reward n]]
    k = bandit.arm_num
    m = bandit.components_num
    para_list = [0]*k
    samples = []
    count_lst = [0]*k
    for i in range(k):
        first_samples = bandit.pull(arm=i+1, size=m)
        para_list[i] = posterior_para(first_samples, bandit)
        for j in range(m):
            samples.append([i + 1, first_samples[j]])
        count_lst[i] += m

    for i in range(m*k, n):
        if i % 200 == 0:
            print('t = ', i)
            # pprint(para_list, indent=2)
        mean_list = [0]*k
        for j in range(k):
            para_arm_j = para_list[j]

            means = para_arm_j['mean']
            mean_sds = para_arm_j['mean_sd']
            dirichlet_concentration = para_arm_j['dirichlet_concentration']

            mean_posterior_list = [float(norm.rvs(loc=means[i], scale=mean_sds[i], size=1)) for i in range(m)]
            dirichlet_sample = np.squeeze(np.random.dirichlet(dirichlet_concentration, size=1))

            weighted_mean = np.dot(mean_posterior_list, dirichlet_sample)
            mean_list[j] = weighted_mean

        action = mean_list.index(max(mean_list))+1
        count_lst[action-1] += 1
        new_x = bandit.pull(arm=action, size=1)[0]
        samples.append([action, new_x])

        prev_samples = np.array([x[1] for x in samples if x[0] == action])
        para_list[action-1] = posterior_para(prev_samples, bandit)

    print(count_lst)
    for dic in para_list:
        for key, value in dic.items():
            print(f'{key:25}{value}')
    return samples
