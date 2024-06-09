import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scienceplots
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info(f"Reading {Config.JSON_PATH} ...")
    data = pd.read_json(Config.JSON_PATH)
    logging.info(f"Data have been read from {Config.JSON_PATH}")
    tree = data[Config.TREE_NAME]

    if not os.path.exists(Config.FOLDER_NAME):
        os.makedirs(Config.FOLDER_NAME)
        logging.info(f"{Config.FOLDER_NAME} folder has been created")

    try:
        for branch_name in Config.BRANCHES_NAME:
            if branch_name in tree:
                branch = tree[branch_name]
                histogram(branch, branch_name)
                logging.info(f"\t{branch_name} histogram was saved in {Config.FOLDER_NAME}/ folder")
    except Exception as exception_info:
        logging.error(exception_info)


def histogram(data, name):
    branch_data = pd.Series(data)
    x_min = branch_data.min()
    x_max = branch_data.max()
    mean_value = branch_data.mean()
    standard_deviation = branch_data.std()
    bins_count = np.histogram_bin_edges(branch_data, bins='scott') # auto, fd, scott

    plt.style.use(['science', 'notebook', 'grid'])
    plt.figure(figsize=(10, 6))
    plt.title(f'Histogram of {name}')
    plt.tick_params(axis='both', labelsize=10)
    #plt.xlabel('')
    plt.ylabel('Number of events')
    plt.hist(branch_data, density=True, histtype='step', bins=bins_count, range=(x_min, x_max), label=name)
    plt.plot([], [], ' ', label=f'Mean: {mean_value:.3f}')
    plt.plot([], [], ' ', label=f'Std Dev: {standard_deviation:.3f}')
    plt.plot([], [], ' ', label=f'Bins: {len(bins_count)}')
    plt.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')
    plt.savefig(f'{Config.FOLDER_NAME}/{name}.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
