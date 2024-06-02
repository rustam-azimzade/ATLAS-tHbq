import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scienceplots
from config import Config


def main():
    data = pd.read_json(Config.JSON_PATH)
    tree = data[Config.TREE_NAME]

    if not os.path.exists(Config.FOLDER_NAME):
        os.makedirs(Config.FOLDER_NAME)

    for branch_name in Config.BRANCHES_NAME:
        if branch_name in tree:
            branch = tree[branch_name]
            histogram(branch, branch_name)


def histogram(data, name):
    branch_data = pd.Series(data)
    x_min = branch_data.min()
    x_max = branch_data.max()
    mean_value = branch_data.mean()
    standard_deviation = branch_data.std()
    bins_count = np.histogram_bin_edges(branch_data, bins='sqrt') # auto, fd, scott, sqrt

    plt.style.use(['science', 'notebook', 'grid'])
    plt.figure(figsize=(10, 6))
    plt.title(f'Histogram of {name}')
    plt.tick_params(axis='both', labelsize=10)
    #plt.xlabel('')
    #plt.xticks(np.linspace(x_min, x_max, num=50))
    plt.xlim(x_min, x_max)
    #plt.ylabel('')
    #plt.ylim(0)
    plt.hist(branch_data, density=True, histtype='step', bins=bins_count, range=(x_min, x_max), label=name)
    plt.plot([], [], ' ', label=f'Mean: {mean_value:.3f}')
    plt.plot([], [], ' ', label=f'Std Dev: {standard_deviation:.3f}')
    plt.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')
    plt.savefig(f'{Config.FOLDER_NAME}/{name}.png', dpi=200)


if __name__ == '__main__':
    main()
