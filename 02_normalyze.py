import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib
import scienceplots
from config import *
import pandas as pd


def main():
    data = pd.read_json('data/json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    branch_name = 'higgs_m'
    data = data[[branch_name]]

    normalized_signal, max_value, min_value = normalize(data)
    transformed_background = normalize(data, max_value, min_value)
    histogram(data[branch_name], normalized_signal[branch_name], branch_name)


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


def histogram(data_before, data_normalized, name):
    mean_value_before = data_before.mean()
    standard_deviation_before = data_before.std()
    bins_count_before = np.histogram_bin_edges(data_before, bins='scott') # auto, fd, scott

    mean_value_normalized = data_normalized.mean()
    standard_deviation_normalized = data_normalized.std()
    bins_count_normalized = np.histogram_bin_edges(data_normalized, bins='scott')  # auto, fd, scott

    plt.style.use(['science', 'notebook', 'grid'])
    matplotlib.rcParams["axes.formatter.limits"] = (-1, 1)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    #axis1.xaxis.set_major_formatter(formatter)
    axis1.xaxis.get_offset_text().set_size(10)
    #axis1.yaxis.set_major_formatter(formatter)
    axis1.yaxis.get_offset_text().set_size(10)
    axis1.set_title(f'Initially')
    axis1.set_ylabel('Number of events')
    axis1.tick_params(axis='both', labelsize=10)
    axis1.hist(data_before, density=True, histtype='step', bins=bins_count_before, label=name)
    axis1.plot([], [], ' ', label=f'Mean: {mean_value_before:.3f}')
    axis1.plot([], [], ' ', label=f'Std Dev: {standard_deviation_before:.3f}')
    axis1.plot([], [], ' ', label=f'Bins: {len(bins_count_before)}')
    axis1.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    #axis2.xaxis.set_major_formatter(formatter)
    axis2.xaxis.get_offset_text().set_size(10)
    #axis2.yaxis.set_major_formatter(formatter)
    axis2.yaxis.get_offset_text().set_size(10)
    axis2.set_title(f'Normalized')
    axis2.tick_params(axis='both', labelsize=10)
    axis2.hist(data_normalized, density=True, histtype='step', bins=bins_count_normalized, label=name)
    axis2.plot([], [], ' ', label=f'Mean: {mean_value_normalized:.3f}')
    axis2.plot([], [], ' ', label=f'Std Dev: {standard_deviation_normalized:.3f}')
    axis2.plot([], [], ' ', label=f'Bins: {len(bins_count_normalized)}')
    axis2.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()

