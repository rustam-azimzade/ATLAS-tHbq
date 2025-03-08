import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib
import scienceplots
from pathlib import Path
from config import Config

FONT_SIZE = 14
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((0, 0))

def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (0, 0),
        'axes.formatter.useoffset': False,
        'axes.formatter.offset_threshold': 1
    })

def main():
    tHbq_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    tt_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')
    ttbb_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_ttbb_SM_300K_(aTTreett;1).json')
    ttH_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_ttH_SM_100K_(aTTreetth;1).json')
    tzbq_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tzbq_SM_100K_(aTTreethbq;1).json')

    signal_data_frame = tHbq_events
    signal_data_frame.index = range(1, len(signal_data_frame) + 1)

    background_data_frame = pd.concat([tt_events, ttbb_events, ttH_events, tzbq_events])
    background_data_frame.index = range(1, len(background_data_frame) + 1)

    folder = Path("../03_results/01_variables_distributions")
    if not folder.exists():
        folder.mkdir()

    for branch_name in Config.VARIABLES_DESCRIPTION.keys():
        signal_variable = signal_data_frame[[branch_name]]
        background_variable = background_data_frame[[branch_name]]

        normalized_signal, max_value, min_value = normalize(signal_variable)
        transformed_background, _, _ = normalize(background_variable, max_value, min_value)

        histogram(signal_variable, background_variable, normalized_signal, transformed_background)


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
    bins_count_signal = np.histogram_bin_edges(signal, bins='scott')

    mean_value_signal_normalized = signal_normalized.mean()
    standard_deviation_signal_normalized = signal_normalized.std()
    bins_count_signal_normalized = np.histogram_bin_edges(signal_normalized, bins='scott')

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    axis1.set_title('Initially')
    axis1.set_ylabel('Number of events', fontsize=FONT_SIZE)
    axis1.xaxis.set_major_formatter(MY_FORMATTER)
    axis1.yaxis.set_major_formatter(MY_FORMATTER)
    axis1.xaxis.get_offset_text().set_size(FONT_SIZE)
    axis1.yaxis.get_offset_text().set_size(FONT_SIZE)
    axis1.tick_params(axis='both', labelsize=FONT_SIZE)
    axis1.hist(background, alpha=0.4, histtype='bar', bins=bins_count_signal, label='Background', color='blue')
    axis1.hist(signal, alpha=0.9, hatch='//', histtype='step',  bins=bins_count_signal, label='Signal', color='red')
    axis1.plot([], [], ' ', label=f'Mean: {mean_value_signal:.3f}')
    axis1.plot([], [], ' ', label=f'Std Dev: {standard_deviation_signal:.3f}')
    axis1.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_title('Normalized')
    axis2.xaxis.set_major_formatter(MY_FORMATTER)
    axis2.yaxis.set_major_formatter(MY_FORMATTER)
    axis2.xaxis.get_offset_text().set_size(FONT_SIZE)
    axis2.yaxis.get_offset_text().set_size(FONT_SIZE)
    axis2.tick_params(axis='both', labelsize=FONT_SIZE)
    axis2.hist(background_transformed, alpha=0.4, histtype='bar', bins=bins_count_signal_normalized,
               label='Background', color='blue')
    axis2.hist(signal_normalized, alpha=0.9, hatch='//', histtype='step', bins=bins_count_signal_normalized,
               label='Signal', color='red')
    axis2.plot([], [], ' ', label=f'Mean: {mean_value_signal_normalized:.3f}')
    axis2.plot([], [], ' ', label=f'Std Dev: {standard_deviation_signal_normalized:.3f}')
    axis2.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    is_abscissa_offset_text = axis1.xaxis.get_offset_text().get_visible()
    if is_abscissa_offset_text:
        abscissa_label_y_position = -0.07
    else:
        abscissa_label_y_position = 0

    figure.text(0.5, abscissa_label_y_position, f'{Config.VARIABLES_DESCRIPTION[name]} ({name})', ha='center',
                fontsize=FONT_SIZE)

    plt.savefig(f'../03_results/01_variables_distributions/{name}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    set_plot_style()
    main()
