import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib
import scienceplots
from config import *
import seaborn as sns


def main():
    signal_data_frame = pd.read_json('data/json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    signal_data_frame.index = range(1, len(signal_data_frame) + 1)
    background_data_frame = pd.read_json('data/json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')
    background_data_frame.index = range(1, len(background_data_frame) + 1)

    plt.figure(figsize=(15, 10))
    sns.set(font_scale=1.4)

    corr_matrix = signal_data_frame.corr()
    corr_matrix = np.round(corr_matrix, 2)
    corr_matrix[np.abs(corr_matrix) < 0.3] = 0

    sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

    plt.title('Correlation Matrix')
    plt.show()


if __name__ == '__main__':
    main()
