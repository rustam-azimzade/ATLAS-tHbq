# Каждое дерево в отдельный файл
# преобразовать типы данных

import numpy as np
from matplotlib import pyplot as plt
from config import *
import pandas as pd
import scienceplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def main():
    data = parse_json('data/json/single_t_weighted.json', Config.NECESSARY_TREE)
    data = data[['chi2_min_higgs_m', 'nbjets']]

    normalized_data = normalize(data)
    standardized_data = standardize(data)

    branch_name = 'chi2_min_higgs_m'
    histogram(data[branch_name], normalized_data[branch_name], standardized_data[branch_name], branch_name)


def parse_json(path, tree):
    with open(path, 'r') as file:
        data = pd.read_json(file)

    data = data[tree].apply(pd.Series)
    return data.transpose()


def normalize(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns, index=data.index)


def standardize(data):
    scaler = StandardScaler()
    standardize_data = scaler.fit_transform(data)
    return pd.DataFrame(standardize_data, columns=data.columns, index=data.index)


def histogram(data_before, data_normalized, data_standardized, name):
    mean_value_before = data_before.mean()
    standard_deviation_before = data_before.std()
    bins_count_before = np.histogram_bin_edges(data_before, bins='scott') # auto, fd, scott

    mean_value_normalized = data_normalized.mean()
    standard_deviation_normalized = data_normalized.std()
    bins_count_normalized = np.histogram_bin_edges(data_normalized, bins='scott')  # auto, fd, scott

    mean_value_standardized = data_standardized.mean()
    standard_deviation_standardized = data_standardized.std()
    bins_count_standardized = np.histogram_bin_edges(data_standardized, bins='scott')  # auto, fd, scott

    plt.style.use(['science', 'notebook', 'grid'])
    figure, axes = plt.subplots(1, 3, figsize=(15, 3.5))

    axis1 = axes[0]
    axis1.set_title(f'Initially')
    axis1.set_ylabel('Number of events')
    axis1.tick_params(axis='both', labelsize=10)
    axis1.hist(data_before, density=True, histtype='step', bins=bins_count_before, label=name)
    axis1.plot([], [], ' ', label=f'Mean: {mean_value_before:.3f}')
    axis1.plot([], [], ' ', label=f'Std Dev: {standard_deviation_before:.3f}')
    axis1.plot([], [], ' ', label=f'Bins: {len(bins_count_before)}')
    axis1.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_title(f'Normalized')
    axis2.tick_params(axis='both', labelsize=10)
    axis2.hist(data_normalized, density=True, histtype='step', bins=bins_count_normalized, label=name)
    axis2.plot([], [], ' ', label=f'Mean: {mean_value_normalized:.3f}')
    axis2.plot([], [], ' ', label=f'Std Dev: {standard_deviation_normalized:.3f}')
    axis2.plot([], [], ' ', label=f'Bins: {len(bins_count_normalized)}')
    axis2.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')

    axis3 = axes[2]
    axis3.set_title(f'Standardized')
    axis3.tick_params(axis='both', labelsize=10)
    axis3.hist(data_standardized, density=True, histtype='step', bins=bins_count_standardized, label=name)
    axis3.plot([], [], ' ', label=f'Mean: {mean_value_standardized:.3f}')
    axis3.plot([], [], ' ', label=f'Std Dev: {standard_deviation_standardized:.3f}')
    axis3.plot([], [], ' ', label=f'Bins: {len(bins_count_standardized)}')
    axis3.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()

