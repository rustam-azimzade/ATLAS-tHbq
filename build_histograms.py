import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scienceplots
import os


def main():
    JSON_PATH = 'single_t_weighted.json'
    FOLDER_NAME = 'histograms'
    TREE_NAME = 'nominal_Loose'
    BRANCHES_NAME = ['nnonbjets', 'foxWolfram_2_momentum', 'chi2_min_higgs_m', 'nbjets', 'njets',
                    'nfwdjets', 'njets_CBT5', 'njets_CBT4', 'sphericity', 'chi2_min_Imvmass_tH',
                    'chi2_min_Whad_m_ttAll', 'chi2_min_tophad_m_ttAll', 'chi2_min_deltaRq1q2',
                    'chi2_min_bbnonbjet_m', 'chi2_min_toplep_pt', 'chi2_min_tophad_pt_ttAll',
                    'bbs_top_m']
    data = pd.read_json(JSON_PATH)
    tree = data[TREE_NAME]

    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)

    for branch_name in BRANCHES_NAME:
        if branch_name in tree:
            branch = tree[branch_name]
            histogram(branch, branch_name)


def histogram(data, name):
    branch_data = pd.Series(data)
    x_min = branch_data.min()
    x_max = branch_data.max()
    mean_value = branch_data.mean()
    bins_count = np.histogram_bin_edges(branch_data, bins='auto') # auto, fd, doane, scott, stone, rice, sturges, sqrt

    plt.style.use(['science', 'notebook', 'grid'])
    plt.figure(figsize=(10, 6))
    plt.hist(branch_data, density=True, histtype='step', bins=bins_count, range=(x_min, x_max), label=name)
    plt.plot([], [], ' ', label=f'Mean: {mean_value:.3f}')
    plt.title(f'Histogram of {name}')
    plt.tick_params(axis='both', labelsize=10)
    #plt.xlabel('')
    #plt.xticks(np.linspace(x_range[0], x_range[1], num=bins_count+1))
    plt.xlim(x_min, x_max)
    plt.ylim(0)
    plt.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')
    plt.savefig(f'histograms/{name}.png', dpi=200)


if __name__ == '__main__':
    main()
