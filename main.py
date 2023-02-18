import os
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

from pyrepo_mcda.mcda_methods import PROMETHEE_II
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs



# plot line sensitivity analysis weights
def plot_sensitivity(vec, data_sust, title = ''):

    # annual rankings chart
    # vec * 100 represents percentage values (in %)
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))


    plt.figure(figsize = (9, 6))
    for j in range(data_sust.shape[0]):

        c = color[j]
        
        plt.plot(vec, data_sust.iloc[j, :], '-', color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(data_sust.index[j], (x_max, data_sust.iloc[j, 0]), # -1 gdy ostatni, 0 gdy pierwsza kolumna bo to kolumna z rankingami
                        fontsize = 14, #style='italic',
                        horizontalalignment='left')


    tit = title.replace('_', ' ')
    if title == 'p_and_q':
        plt.xlabel('Step of modification ' + tit, fontsize = 14)
    else:
        plt.xlabel('Threshold ' + tit + ' in rate of a standard deviation of criteria performances', fontsize = 14)
    plt.ylabel("Rank", fontsize = 14)
    # xlabels = ['{:.0f}'.format(v) for v in vec * 100]
    # plt.xticks(ticks=vec * 100, fontsize = 12)
    plt.xticks(ticks=vec, fontsize = 14)
    plt.yticks(fontsize = 14)
    # plt.gca().invert_xaxis()
    # plt.title(title)
    plt.grid(linestyle = ':')
    plt.tight_layout()
    plt.savefig('./results/' + 'sensitivity' + '_' + title + '.png')
    plt.show()



def main():

    # Criteria indexes of each main dimension.
    modules_indexes = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21, 22, 23]
    ]


    # scenarios for 3D sensitivity analysis of thresholds: 
    # 1 - q
    # 2 - p
    # 3 - p_and_q
    scenario = 'p_and_q'
    
    path = 'DATASET'
    # Number of countries
    m = 16
    # Number of criteria
    n = 24

    file = 'data_2021' + '.csv'
    pathfile = os.path.join(path, file)
    data = pd.read_csv(pathfile, index_col = 'Country')

    country_names = pd.read_csv('./DATASET/country_names_1.csv', index_col = 'Ai')
    list_alt_names_latex = list(country_names.index)

    list_crits_names = [r'$C_{' + str(i) + '}$' for i in range(1, data.shape[1] + 1)]
    
    results = pd.DataFrame(index = list_alt_names_latex)


    if scenario == 'q':
        # scenario 1 q
        # 11 steps
        thresholds = np.linspace(0, 0.5, 9)

    elif scenario == 'p':
        # scenario 2 p
        thresholds = np.linspace(0, 2, 9)

    elif scenario == 'p_and_q':
        # scenario 3
        thresholds = np.linspace(0, 0.5, 9) # q (thresholds represents q here)
        thresholds_p = np.linspace(0, 2, 9) # p


    # matrix
    matrix = data.to_numpy()

    # types healthcare
    types = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1])

    # weights
    weights = mcda_weights.critic_weighting(matrix)

    # initialization of PROMETHEE II with default parameters: p, q, usual preference functions
    promethee_II = PROMETHEE_II()
    preference_functions = [promethee_II._linear_function for pf in range(len(weights))]

    u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
    p = 2 * u
    q = 0.5 * u

    pref = promethee_II(matrix, weights, types, preference_functions=preference_functions, p = p, q = q)
    rank = rank_preferences(pref, reverse=True)

    results_base = pd.DataFrame(index = list_alt_names_latex)
    results_base['Preference'] = pref
    results_base['Rank'] = rank
    results_base.to_csv('./results/results_base.csv')

    
    for el_p, thr in enumerate(thresholds):

        # initialization of PROMETHEE II with default parameters: p, q, usual preference functions
        promethee_II = PROMETHEE_II()
        preference_functions = [promethee_II._linear_function for pf in range(len(weights))]

        u = np.sqrt(np.sum(np.square(np.mean(matrix, axis = 0) - matrix), axis = 0) / matrix.shape[0])
        p = 2 * u
        q = 0.5 * u

        if scenario == 'q':
            # scenario 1
            # q[modules_indexes[0]] = thr * u[modules_indexes[0]]
            q = thr * u

        elif scenario == 'p':
            # scenario 2
            # p[modules_indexes[0]] = thr * u[modules_indexes[0]]
            p = thr * u

        if scenario == 'p_and_q':
            # scenario 3
            # p[modules_indexes[0]] = thresholds_p[el_p] * u[modules_indexes[0]]
            # q[modules_indexes[0]] = thr * u[modules_indexes[0]]

            p = thresholds_p[el_p] * u
            q = thr * u
        
        pref = promethee_II(matrix, weights, types, preference_functions=preference_functions, p = p, q = q)
        rank = rank_preferences(pref, reverse=True)
        results[str(el_p)] = rank

    if scenario == 'p_and_q':
        thresholds = np.arange(1, 10)

    plot_sensitivity(np.round(thresholds, 2), results, scenario)


if __name__ == '__main__':
    main()