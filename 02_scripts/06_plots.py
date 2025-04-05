import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import ScalarFormatter
from torchgen.executorch.api.et_cpp import return_names
from sklearn.model_selection import train_test_split

from config import Config
import tensorflow as tf
import random
import os

WEIGHTS_SEED_NUMBER = 35
GLOBAL_SEED_NUMBER = 5
FONT_SIZE = 14
PLOTS_SAVE_PATH = '../03_results/03_neural_network/01_performance_plots'

MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((0, 0))

np.random.seed(GLOBAL_SEED_NUMBER)
tf.random.set_seed(GLOBAL_SEED_NUMBER)
random.seed(GLOBAL_SEED_NUMBER)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'


WEIGHTS = {
    "tHbq": 2.0,
    "tt": 1.0,
    "ttbb": 1.862340,
    "ttH": 0.268164,
    "tZbq": 0.085833
}

SIGNAL_SIGNIFICANCE_WEIGHTS = {
    "tHbq": 0.00932265,
    "tt": 0.0537442,
    "ttbb": 0.100090,
    "ttH": 0.0144123,
    "tZbq": 0.00461306
}


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.useoffset': False,
        'axes.formatter.offset_threshold': 1
    })


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


def load_data():
    tHbq_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    tt_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')
    ttbb_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_ttbb_SM_300K_(aTTreett;1).json')
    ttH_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_ttH_SM_100K_(aTTreetth;1).json')
    tZbq_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tzbq_SM_100K_(aTTreethbq;1).json')

    for branch_name in Config.VARIABLES_DESCRIPTION:
        tHbq_events[branch_name], max_value, min_value = normalize(tHbq_events[branch_name])
        tt_events[branch_name], _, _ = normalize(tt_events[branch_name], max_value, min_value)
        ttbb_events[branch_name], _, _ = normalize(ttbb_events[branch_name], max_value, min_value)
        ttH_events[branch_name], _, _ = normalize(ttH_events[branch_name], max_value, min_value)
        tZbq_events[branch_name], _, _ = normalize(tZbq_events[branch_name], max_value, min_value)

    # Label data
    tHbq_events['signal'] = 1
    tt_events['signal'] = 0
    ttbb_events['signal'] = 0
    ttH_events['signal'] = 0
    tZbq_events['signal'] = 0

    tHbq_events['weight'] = WEIGHTS['tHbq']
    tt_events['weight'] = WEIGHTS['tt']
    ttbb_events['weight'] = WEIGHTS['ttbb']
    ttH_events['weight'] = WEIGHTS['ttH']
    tZbq_events['weight'] = WEIGHTS['tZbq']

    tHbq_events['significance_weight'] = SIGNAL_SIGNIFICANCE_WEIGHTS['tHbq']
    tt_events['significance_weight'] = SIGNAL_SIGNIFICANCE_WEIGHTS['tt']
    ttbb_events['significance_weight'] = SIGNAL_SIGNIFICANCE_WEIGHTS['ttbb']
    ttH_events['significance_weight'] = SIGNAL_SIGNIFICANCE_WEIGHTS['ttH']
    tZbq_events['significance_weight'] = SIGNAL_SIGNIFICANCE_WEIGHTS['tZbq']

    # Prepare data
    total_events = pd.concat([tHbq_events, tt_events, ttbb_events, ttH_events, tZbq_events])
    total_events = total_events.sample(frac=1).reset_index(drop=True)
    total_events.index = range(1, len(total_events) + 1)

    input_data = total_events.drop(columns=['signal', 'weight', 'significance_weight'])
    output_data = pd.Series(total_events['signal'])
    events_weights = total_events['weight']
    significance_weights = total_events['significance_weight']

    return input_data, output_data, events_weights, significance_weights


def calculate_separation_power(signal_predictions, background_predictions, signal_weights, background_weights):
    bin_edges = np.linspace(0, 1, 40)
    signal_hist, _ = np.histogram(signal_predictions, bins=bin_edges, weights=signal_weights, density=True)
    background_hist, _ = np.histogram(background_predictions, bins=bin_edges, weights=background_weights, density=True)

    signal_hist /= np.sum(signal_hist)
    background_hist /= np.sum(background_hist)

    separation_power = 0
    for i in range(len(signal_hist)):
        if signal_hist[i] == 0 and background_hist[i] == 0:
            continue
        separation_power += (signal_hist[i] - background_hist[i]) ** 2 / (signal_hist[i] + background_hist[i])
    separation_power *= 0.5
    return separation_power


def calculate_signal_significance(signal_predictions, background_predictions, signal_weights, background_weights, threshold):
    bin_edges = np.linspace(0, 1, 40)
    signal_hist, _ = np.histogram(signal_predictions, bins=bin_edges, weights=signal_weights, density=False)
    background_hist, _ = np.histogram(background_predictions, bins=bin_edges, weights=background_weights, density=False)

    # Нормировка площади гистограммы на единицу
    #signal_hist /= np.sum(signal_hist)
    #background_hist /= np.sum(background_hist)

    significance = 0
    threshold_index = np.digitize(threshold, bin_edges)

    selected_signal = signal_hist[threshold_index:]
    selected_background = background_hist[threshold_index:]
    S = np.sum(selected_signal)
    B = np.sum(selected_background)

    if S > 0 and B > 0:
        significance = S / np.sqrt(S + B)
    else:
        significance = 0.0

    return significance


def get_all_signal_significance(signal_predictions, background_predictions, signal_significance_weights, background_significance_weights):
    significances = []
    max_significance = 0.0
    optimal_threshold = None

    bin_edges = np.linspace(0, 1, 40)
    for threshold in bin_edges:
        significance = calculate_signal_significance(signal_predictions, background_predictions, signal_significance_weights, background_significance_weights, threshold)
        if significance > max_significance:
            max_significance = significance
            optimal_threshold = threshold
        significances.append(significance)
    return significances, max_significance, optimal_threshold


def save_roc_curve(predictions, outputs, weights):
    fpr, tpr, thresholds = roc_curve(outputs, predictions, sample_weight=weights, drop_intermediate=False)
    auc_score = auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic', fontsize=FONT_SIZE)
    plt.ylabel('True Positive Rate', fontsize=FONT_SIZE)
    plt.xlabel('False Positive Rate', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    plt.savefig(f'{PLOTS_SAVE_PATH}/01_png/roc_curve.png', dpi=300)
    plt.savefig(f'{PLOTS_SAVE_PATH}/02_pdf/roc_curve.pdf',)
    plt.close()


def save_histogram_of_predictions(signal_predictions, background_predictions, signal_weights, background_weights, signal_significance_weights, background_significance_weights):
    bin_edges = np.linspace(0, 1, 40)

    _, max_significance, _ = get_all_signal_significance(signal_predictions, background_predictions, signal_significance_weights, background_significance_weights)
    separation_power = calculate_separation_power(signal_predictions, background_predictions, signal_weights, background_weights)

    plt.figure()
    plt.title('Histogram of Neural Network Output', fontsize=FONT_SIZE)
    plt.xlabel('Predicted Probability', fontsize=FONT_SIZE)
    plt.ylabel('Number of events', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.hist(signal_predictions, bins=bin_edges, alpha=0.9, hatch='//', histtype='step', label='Signal (pp → tH)',
             color='red', weights=signal_weights)
    plt.hist(background_predictions, bins=bin_edges, alpha=0.4, label='Background', color='blue', weights=background_weights)
    plt.legend(loc='upper center', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')
    plt.annotate(f'Separation Power: {separation_power * 100:.2f}%\nSignal Significance: {max_significance:.2f}',
                 xy=(0.3, 0.80), xycoords='axes fraction', fontsize=FONT_SIZE, verticalalignment='top',
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1))

    plt.savefig(f'{PLOTS_SAVE_PATH}/01_png/prediction.png', dpi=300)
    plt.savefig(f'{PLOTS_SAVE_PATH}/02_pdf/prediction.pdf')
    plt.close()


def save_significances(signal_predictions, background_predictions, signal_significance_weights, background_significance_weights):
    bin_edges = np.linspace(0, 1, 40)
    significances, max_significance, optimal_threshold = get_all_signal_significance(signal_predictions, background_predictions, signal_significance_weights, background_significance_weights)

    plt.figure()
    plt.plot(bin_edges, significances, color='blue')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Best threshold = {optimal_threshold:.3f}')
    plt.title('Signal Significance vs Threshold', fontsize=FONT_SIZE)
    plt.xlabel('Classification Threshold', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.ylabel('Signal Significance', fontsize=FONT_SIZE)
    plt.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    plt.savefig(f'{PLOTS_SAVE_PATH}/01_png/significances.png', dpi=300)
    plt.savefig(f'{PLOTS_SAVE_PATH}/02_pdf/significances.pdf')
    plt.close()


def main():
    input_data, output_data, events_weights, significance_weights = load_data()

    _, input_test, _, output_test, _, weights_test, _, significance_weights_test = train_test_split(
        input_data, output_data, events_weights, significance_weights,
        test_size=0.3,
        shuffle=True,
        random_state=GLOBAL_SEED_NUMBER,
        stratify=output_data
    )

    neural_network = load_model('../03_results/03_neural_network/02_pre-trained_model/tH(bb)_signal_classification.hdf5')
    output_predicted = neural_network.predict(input_test).ravel()

    signal_mask = output_test == 1
    background_mask = output_test == 0

    signal_weights = weights_test[signal_mask]
    background_weights = weights_test[background_mask]
    signal_significance_weights = significance_weights_test[signal_mask]
    background_significance_weights = significance_weights_test[background_mask]

    signal_predictions = output_predicted[signal_mask]
    background_predictions = output_predicted[background_mask]

    save_roc_curve(output_predicted, output_test, weights_test)
    save_histogram_of_predictions(signal_predictions, background_predictions, signal_weights, background_weights, signal_significance_weights, background_significance_weights)
    save_significances(signal_predictions, background_predictions, signal_significance_weights, background_significance_weights)

if __name__ == '__main__':
    set_plot_style()
    main()