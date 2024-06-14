import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib
import scienceplots
from config import *


def main():
    signal_data_frame = pd.read_json('data/json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    signal_data_frame.index = range(1, len(signal_data_frame) + 1)
    background_data_frame = pd.read_json('data/json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')
    background_data_frame.index = range(1, len(background_data_frame) + 1)

    branch_name = 'higgs_m'
    signal_data_frame = signal_data_frame[[branch_name]]
    background_data_frame = background_data_frame[[branch_name]]

    normalized_signal, max_value, min_value = normalize(signal_data_frame)
    transformed_background, _, _ = normalize(background_data_frame, max_value, min_value)
    histogram(signal_data_frame, background_data_frame, normalized_signal, transformed_background)


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


def histogram(signal, background, signal_normalized, background_transformed):
    name = signal.columns[0]

    signal = signal.squeeze()
    background = background.squeeze()
    signal_normalized = signal_normalized.squeeze()
    background_transformed = background_transformed.squeeze()

    mean_value_signal = signal.mean()
    standard_deviation_signal = signal.std()
    bins_count_signal = np.histogram_bin_edges(signal, bins='scott') # auto, fd, scott

    mean_value_signal_normalized = signal_normalized.mean()
    standard_deviation_signal_normalized = signal_normalized.std()
    bins_count_signal_normalized = np.histogram_bin_edges(signal_normalized, bins='scott')

    plt.style.use(['science', 'notebook', 'grid'])
    matplotlib.rcParams["axes.formatter.limits"] = (-1, 1)
    matplotlib.rcParams['axes.formatter.useoffset'] = True
    matplotlib.rcParams['axes.formatter.offset_threshold'] = 1
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    axis1.set_title('Initially')
    axis1.set_ylabel('Number of events', fontsize=10)
    axis1.xaxis.set_major_formatter(formatter)
    axis1.yaxis.set_major_formatter(formatter)
    axis1.xaxis.get_offset_text().set_size(10)
    axis1.yaxis.get_offset_text().set_size(10)
    axis1.tick_params(axis='both', labelsize=10)
    axis1.hist(signal, density=False, histtype='step', bins=bins_count_signal, label='Signal', color='blue')
    axis1.plot([], [], ' ', label=f'Mean: {mean_value_signal:.3f}')
    axis1.plot([], [], ' ', label=f'Std Dev: {standard_deviation_signal:.3f}')
    axis1.hist(background, density=False, histtype='step', bins=bins_count_signal, label='Background', color='red')
    axis1.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_title('Normalized')
    axis2.xaxis.set_major_formatter(formatter)
    axis2.xaxis.get_offset_text().set_size(10)
    axis2.yaxis.set_major_formatter(formatter)
    axis2.yaxis.get_offset_text().set_size(10)
    axis2.tick_params(axis='both', labelsize=10)
    axis2.hist(signal_normalized, density=False, histtype='step', bins=bins_count_signal_normalized, label='Signal', color='blue')
    axis2.plot([], [], ' ', label=f'Mean: {mean_value_signal_normalized:.3f}')
    axis2.plot([], [], ' ', label=f'Std Dev: {standard_deviation_signal_normalized:.3f}')
    axis2.hist(background_transformed, density=False, histtype='step', bins=bins_count_signal_normalized, label='Background', color='red')
    axis2.legend(loc='upper right', fontsize=10, fancybox=False, edgecolor='black')

    figure.text(0.5, -0.05, f'{Config.VARIABLES_DESCRIPTION[name]} ({name})', ha='center', fontsize=10)

    plt.show()
    #plt.savefig(f'{name}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
