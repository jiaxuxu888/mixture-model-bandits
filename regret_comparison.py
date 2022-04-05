from Thompson_Sampling import *
from UCB_Normal import *
from UCB_V import *
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


def regret_comparison(bandit, algorithm_list, n, rep, folder_name, show: bool):
    # function to compare regrets and save or display one plot,
    # also saving the parameters to a txt file.

    # bandit: a bandit instance, either gaussian or beta
    # algorithm_list: [[algorithm, algorithm name], ...]
    # n: horizon
    # rep: how many repeated experiments for this bandit problem?
    # folder_name: folder name
    # show: boolean, show the plot immediately or save it?

    print(bandit.parameters(latex=False))
    [optimal_mean, optimal_action] = [bandit.optimal_arm_mean, bandit.optimal_arm]
    optimal_samples = [(i + 1) * optimal_mean for i in range(n)]
    print(f"The optimal action is the arm {optimal_action}. \n\n")

    time_record = []
    regret_df = pd.DataFrame({})
    for [algorithm, algorithm_name] in algorithm_list:
        print(f" {algorithm_name} ".center(120, '#'))
        print(bandit.parameters(latex=False))
        total_time = 0

        for i in range(rep):
            print('{0}, {1:1d}th iteration'.format(algorithm_name, i + 1))

            tic = time.perf_counter()
            samples = algorithm(bandit, n)
            toc = time.perf_counter()
            total_time += toc - tic

            pseudo_means = [bandit.weighted_mean[x[0] - 1] for x in samples]
            regret = np.array(optimal_samples) - np.cumsum(pseudo_means)
            regret = pd.DataFrame({'Cumulative Regret': regret,
                                   'Algorithm / Time': algorithm_name,
                                   't': np.arange(n) + 1})
            regret_df = pd.concat([regret_df, regret], axis=0)

            print(f"Time: {toc - tic:0.3f} secs, approximately remaining:"
                  f" {total_time / (i + 1) * (rep - i - 1) / 60:0.2f} minutes. \n")

        time_record.append(total_time / rep)

    for i in range(len(time_record)):
        print(f"{algorithm_list[i][1]} on average {time_record[i]:0.3f} secs per experiment.")
        name_with_time = f"{algorithm_list[i][1]} / {time_record[i]:0.3f}"
        regret_df.loc[regret_df['Algorithm / Time'] == algorithm_list[i][1],
                      'Algorithm / Time'] = name_with_time

    if bandit.is_beta:
        [alpha1, alpha2] = bandit.alpha_range
        [beta1, beta2] = bandit.beta_range
        name_info = f"k = {bandit.arm_num:d},  m = {bandit.components_num:d},  n = {n:d},  r = {rep:d}"
        name_identify = f"{name_info}, [{alpha1}, {alpha2}], [{beta1}, {beta2}]"
        name_without_suffix = name_identify
    else:
        [mean1, mean2] = bandit.mean_range
        [sd1, sd2] = bandit.sd_range
        name_info = f"k = {bandit.arm_num:d},  m = {bandit.components_num:d},  n = {n:d},  r = {rep:d}"
        name_identify = f"{name_info},  [{mean1}, {mean2}]"
        name_without_suffix = name_identify

    print('\n\nSaving parameters......\n\n')

    try:
        os.mkdir(os.path.join(os.getcwd(), folder_name))
        os.mkdir(os.path.join(os.getcwd(), folder_name, 'parameters'))
        os.mkdir(os.path.join(os.getcwd(), folder_name, 'plots'))
    except FileExistsError:
        pass

    my_int = 1
    while os.path.exists(os.path.join(os.getcwd(), folder_name, 'parameters', name_identify)):
        name_identify = f"{name_without_suffix} ({my_int})"
        my_int += 1

    my_directory_para = os.path.join(os.getcwd(), folder_name, 'parameters', name_identify)
    my_directory_png = os.path.join(os.getcwd(), folder_name, 'plots', name_identify)
    f = open(my_directory_para, 'w+')
    f.write(bandit.parameters(latex=True))
    f.close()

    print('Saving plots......\n\n')
    sns.set(style="whitegrid", palette="muted", color_codes=True)
    sns.lineplot(data=regret_df, x="t", y='Cumulative Regret', hue='Algorithm / Time')
    plt.rcParams.update({'font.size': 100})

    if bandit.is_beta:
        plt.title(f"{name_info},  ${alpha1} < a < {alpha2}$,  ${beta1} < b < {beta2}$",
                  fontsize=14)
    else:
        plt.title(f"{name_info},  ${mean1} < \mu < {mean2}$,  ${sd1} < \sigma < {sd2}$",
                  fontsize=14)
    plt.savefig(f"{my_directory_png}.png")
    print(' DONE! '.center(120, '#')+'\n\n')
    if show:
        plt.show()
    plt.clf()


def make_plots(k_lst, mean_range_lst, n_lst, components_num, rep, folder_name, algorithm_lst, plots_num=1):
    # function for drawing and saving multiple plots and parameters

    # k_lst: a list containing the number of arms an environment carries
    # components_lst: a list containing possible numbers of components
    # n_lst: a list containing horizons
    # mean_range_list: list of the form [[a1, b1], ..., [an, bn]] denoting the range of means
    # rep: how many experiments to repeat under a given environment
    # folder_name: folder name
    # algorithm_lst: which algorithms to test on? list of the form [[algorithm, name], ...]
    # plots_num: repeat all for how many plots?

    for a_plot in range(plots_num):
        for k in k_lst:
            for index in range(len(mean_range_lst)):
                for m in components_num:
                    print(f"k = {k}, m = {m}, mean range is {mean_range_lst[index]}".center(130, '@'))
                    bandit_1 = MixtureBandit(arm_num=k,
                                             components_num=m,
                                             mean_range=mean_range_lst[index],
                                             sd_range=[0.5, 1])
                    regret_comparison(bandit_1,
                                      n=n_lst[index],
                                      rep=rep,
                                      folder_name=folder_name,
                                      show=False,
                                      algorithm_list=algorithm_lst)


def algo_lst(index):
    algo = [[ucb_v, 'UCB-V'],                                # 1
            [ucb_v_em, 'UCB-V-EM'],                          # 2
            [ucb_normal, 'UCB1-Normal'],                     # 3
            [ucb_normal_em, 'UCB1-Normal-EM'],               # 4
            [thompson_sampling, 'Thompson Sampling'],        # 5
            [ucb, 'UCB'],                                    # 6
            [ucb_em, 'UCB-EM']]                              # 7
    return [algo[i-1] for i in index]


make_plots(k_lst=[2, 4, 8],
           mean_range_lst=[[0, 1], [0, 5]],
           n_lst=[1000, 500],
           components_num=[2, 3, 4],
           rep=20,
           folder_name='Thompson_sampling',
           algorithm_lst=algo_lst([2, 5]))
