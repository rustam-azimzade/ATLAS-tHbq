import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import roc_curve, auc
from tensorflow.python.checkpoint.tensor_callable import Callable
import logging
from .config import Config
from pathlib import Path
import time
import os


logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    level=logging.INFO
)
logging.getLogger('fontTools').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class Plots:
    def __init__(self, save_path='/', formatter=Config.MY_FORMATTER):
        self._save_path = Path(save_path)
        self._formatter = formatter

        self._current_figure = None
        self._current_plot_name = None

        self._set_plot_style()
        self._create_save_dirs()


    def _set_plot_style(self):
        plt.style.use(['science', 'notebook', 'grid'])
        plt.rcParams.update({
            'font.size': Config.FONT_SIZE,
            'axes.labelsize': Config.FONT_SIZE,
            'axes.titlesize': Config.FONT_SIZE + 2,
            'legend.fontsize': Config.FONT_SIZE - 1,
            'xtick.labelsize': Config.FONT_SIZE - 1,
            'ytick.labelsize': Config.FONT_SIZE - 1,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.formatter.useoffset': False,
            'axes.formatter.offset_threshold': 1,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.autolayout': True,
        })


    def _create_save_dirs(self):
        os.makedirs(os.path.join(self._save_path, '01_png'), exist_ok=True)
        os.makedirs(os.path.join(self._save_path, '02_pdf'), exist_ok=True)


    @staticmethod
    def test_time(function):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = function(*args, **kwargs)
            end_time = time.time()
            work_time = end_time - start_time
            logger.info(f'Working time: {work_time:.6f} seconds')
            return result
        return wrapper


    @staticmethod
    def debug_separation_power(function):
        def wrapper(*args, **kwargs):
            variable_name = kwargs.get('variable_name', None)
            logger.info(f'Start debugging of the separation power calculation for variable {variable_name}')
            debug_data = function(*args, **kwargs, debug=True)
            (signal_hist, background_hist,
            signal_hist_weighted, background_hist_weighted,
            signal_hist_weighted_normalized, background_hist_weighted_normalized,
            bins_edges) = debug_data
            # Displaying the table header
            print(f"{'Bin':>3} | {'Edge_low':>10} | {'Edge_high':>10} | {'S':>8} | {'B':>8} | {'S_w':>10} | "
                  f"{'B_w':>14} | {'S_norm':>12} | {'B_norm':>12} | {'SP contrib':>12} | {'SP tot':>12}")
            print("-" * 140)

            SP_contributions = list()
            separation_power = 0.0

            for i in range(len(signal_hist)):
                S = signal_hist[i]
                B = background_hist[i]
                S_weighted = signal_hist_weighted[i]
                B_weighted = background_hist_weighted[i]
                S_weighted_normalized = signal_hist_weighted_normalized[i]
                B_weighted_normalized = background_hist_weighted_normalized[i]

                if (S_weighted_normalized + B_weighted_normalized) > 0:
                    SP_contribution = 0.5 * ((S_weighted_normalized - B_weighted_normalized) ** 2) / (S_weighted_normalized + B_weighted_normalized)
                else:
                    SP_contribution = 0.0
                separation_power += SP_contribution
                SP_contributions.append(SP_contribution)

                # Печать строки с данными
                print(f"{i:3d} | {bins_edges[i]:10.4f} | {bins_edges[i + 1]:10.4f} | {S:8.0f} | {B:8.0f} | "
                      f"{S_weighted:10.3f} | {B_weighted:14.3f} | {S_weighted_normalized:12.6f} | "
                      f"{B_weighted_normalized:12.6f} | {SP_contribution:12.6f} | "
                      f"{separation_power:12.6f}")
            print("-" * 140)
            print(f"{'Total separation power:':>72} {separation_power:12.6f}")
            return separation_power
        return wrapper


    @test_time
    @debug_separation_power
    def _calculate_separation_power(self, signal_variable_values, background_variable_values, bins_edges, signal_variable_weights=None, background_variable_weights=None, variable_name=None, debug=False):
        signal_variable_values = np.asarray(signal_variable_values)
        background_variable_values = np.asarray(background_variable_values)
        bins_edges = np.asarray(bins_edges)

        if signal_variable_weights is None:
            signal_variable_weights = np.ones_like(signal_variable_values)
        else:
            signal_variable_weights = np.asarray(signal_variable_weights)
        if background_variable_weights is None:
            background_variable_weights = np.ones_like(background_variable_values)
        else:
            background_variable_weights = np.asarray(background_variable_weights)

        signal_hist, _ = np.histogram(signal_variable_values, bins=bins_edges, density=False)
        background_hist, _ = np.histogram(background_variable_values, bins=bins_edges, density=False)

        signal_hist_weighted, _ = np.histogram(signal_variable_values, bins=bins_edges, weights=signal_variable_weights, density=False)
        background_hist_weighted, _ = np.histogram(background_variable_values, bins=bins_edges, weights=background_variable_weights, density=False)

        # Normalization
        signal_hist_weighted_sum = np.sum(signal_hist_weighted)
        background_hist_weighted_sum = np.sum(background_hist_weighted)
        signal_hist_weighted_normalized = signal_hist_weighted / signal_hist_weighted_sum
        background_hist_weighted_normalized = background_hist_weighted / background_hist_weighted_sum

        if debug:
            debug_data = (signal_hist, background_hist,
                          signal_hist_weighted, background_hist_weighted,
                          signal_hist_weighted_normalized, background_hist_weighted_normalized,
                          bins_edges)
            return debug_data

        nonzero_mask = (signal_hist_weighted_normalized + background_hist_weighted_normalized) > 0
        if np.any(nonzero_mask):
            separation_power = 0.5 * np.sum((signal_hist_weighted_normalized[nonzero_mask] - background_hist_weighted_normalized[nonzero_mask]) ** 2 / (signal_hist_weighted_normalized[nonzero_mask] + background_hist_weighted_normalized[nonzero_mask]))
        else:
            separation_power = 0.0

        logger.info(f"Separation Power for variable {variable_name} is calculated")

        return separation_power


    def plot_histogram(self, signal_events, background_events, signal_weights=None, background_weights=None, normalize_to_background: bool = False):
        if signal_weights is None:
            signal_weights = np.ones_like(signal_events)
        if background_weights is None:
            background_weights = np.ones_like(background_events)

        variable_name = signal_events.columns[0]

        signal_data = signal_events[variable_name].copy().squeeze()
        background_data = background_events[variable_name].copy().squeeze()

        binning = Config.BINNING_RANGES.get(variable_name)
        if binning:
            num_bins, start, stop = binning
            bins_edges = np.linspace(start, stop, num_bins + 1)
        else:
            bins_edges = np.histogram_bin_edges(signal_data, bins='scott')
        separation_power = self._calculate_separation_power(
            signal_variable_values=signal_data, background_variable_values=background_data, bins_edges=bins_edges,
            signal_variable_weights=signal_weights, background_variable_weights=background_weights,
            variable_name=variable_name
        )
        #mean_signal = np.average(signal_data, weights=signal_weights)
        #std_signal = np.sqrt(
        #    np.average((signal_data - mean_signal) ** 2, weights=signal_weights)
        #)
        y_label_name = 'Number of Events'
        if normalize_to_background:
            total_background_weight = np.sum(background_weights)
            total_signal_weight = np.sum(signal_weights)
            if total_signal_weight > 0:
                signal_weights *= (total_background_weight / total_signal_weight)

            y_label_name = 'Normalized Number of Events'

        figure, axis = plt.subplots(figsize=Config.PLOT_SIZE)
        axis.xaxis.set_major_formatter(self._formatter)
        axis.yaxis.set_major_formatter(self._formatter)
        axis.xaxis.get_offset_text().set_size(Config.FONT_SIZE)
        axis.yaxis.get_offset_text().set_size(Config.FONT_SIZE)
        axis.tick_params(axis='both', labelsize=Config.FONT_SIZE)
        axis.ticklabel_format(axis='both', style='sci', scilimits=Config.POWER_LIMITS)
        axis.hist(background_data, weights=background_weights, alpha=0.4, histtype='bar', bins=bins_edges, label='Background', color='blue')
        axis.hist(signal_data, weights=signal_weights, alpha=0.9, hatch='//', histtype='step', bins=bins_edges, label='Signal', color='red')
        #axis.plot([], [], ' ', label=f'Mean: {mean_signal:.3f}')
        #axis.plot([], [], ' ', label=f'Std Dev: {std_signal:.3f}')
        axis.annotate(f'Separation Power: {separation_power * 100:.2f}%',
                      xy=(0.05, 0.95), xycoords='axes fraction', fontsize=Config.FONT_SIZE, verticalalignment='top',
                      bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1))
        axis.legend(loc='center right', fontsize=Config.FONT_SIZE, fancybox=False, edgecolor='black')
        axis.set_ylabel(y_label_name, fontsize=Config.FONT_SIZE)
        is_abscissa_offset_text = axis.xaxis.get_offset_text().get_visible()
        abscissa_label_y_position = -0.04 if is_abscissa_offset_text else 0.0
        figure.text(0.5, abscissa_label_y_position, f'{Config.VARIABLES_DESCRIPTION[variable_name]} ({variable_name})', ha='center',
                    fontsize=Config.FONT_SIZE)

        # Store in the class fields
        self._current_figure = figure
        self._current_plot_name = variable_name

        return figure


    def save_plot(self, plot_name=None):
        if self._current_figure is None:
            raise RuntimeError("No figure to save. Generate a plot first.")

        name = plot_name or self._current_plot_name
        if name is None:
            raise ValueError("Plot name must be specified.")

        png_path = os.path.join(self._save_path, '01_png', f'{name}.png')
        pdf_path = os.path.join(self._save_path, '02_pdf', f'{name}.pdf')

        self._current_figure.savefig(png_path, dpi=300)
        self._current_figure.savefig(pdf_path)
        plt.close(self._current_figure)

        self._current_figure = None
        self._current_plot_name = None


    def plot_roc_curve(self, predictions, true_labels, weights):
        fpr, tpr, _ = roc_curve(true_labels, predictions, sample_weight=weights, drop_intermediate=False)
        auc_score = auc(fpr, tpr)

        figure, axis = plt.subplots(figsize=Config.PLOT_SIZE)
        axis.plot(fpr, tpr, color='blue', label=f'ROC-curve (AUC = {auc_score:.4f})')
        axis.plot([0, 1], [0, 1], color='red', linestyle='--')
        axis.set_title('Receiver Operating Characteristic', fontsize=Config.FONT_SIZE)
        axis.set_ylabel('True Positive Rate', fontsize=Config.FONT_SIZE)
        axis.set_xlabel('False Positive Rate', fontsize=Config.FONT_SIZE)
        axis.tick_params(axis='both', labelsize=Config.FONT_SIZE)
        axis.ticklabel_format(axis='both', style='sci', scilimits=Config.POWER_LIMITS)
        axis.legend(loc='best', fontsize=Config.FONT_SIZE, fancybox=False, edgecolor='black')

        self._current_figure = figure
        self._current_plot_name = 'ROC-Curve'

        return figure


    def save_prediction_histogram(self, signal_predictions, background_predictions, signal_weights, background_weights):
        separation_power = self._calculate_separation_power(signal_predictions, background_predictions, signal_weights, background_weights)

        figure, axis = plt.subplots(figsize=Config.PLOT_SIZE)
        axis.title('Histogram of Neural Network Output', fontsize=Config.FONT_SIZE)
        axis.xlabel('Predicted Probability', fontsize=Config.FONT_SIZE)
        axis.ylabel('Weighted Number of Events', fontsize=Config.FONT_SIZE)
        axis.tick_params(axis='both', labelsize=Config.FONT_SIZE)
        axis.ticklabel_format(axis='both', style='sci', scilimits=Config.POWER_LIMITS)
        axis.hist(signal_predictions, bins=Config.BIN_EDGES, alpha=0.9, hatch='//', histtype='step', label='Signal (pp → tH)',
                 color='red', weights=signal_weights)
        axis.hist(background_predictions, bins=Config.BIN_EDGES, alpha=0.4, label='Background', color='blue',
                 weights=background_weights)
        axis.legend(loc='upper center', fontsize=Config.FONT_SIZE, fancybox=False, edgecolor='black')
        axis.annotate(f'Separation Power: {separation_power * 100:.2f}%',
                     xy=(0.3, 0.80), xycoords='axes fraction', fontsize=Config.FONT_SIZE, verticalalignment='top',
                     bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
                     )

        self._current_figure = figure
        self._current_plot_name = 'prediction'

        return figure


    def _calculate_signal_significance(self, signal_predictions, background_predictions, signal_weights, background_weights, threshold):
        signal_hist, _ = np.histogram(signal_predictions, bins=Config.BIN_EDGES, weights=signal_weights, density=False)
        background_hist, _ = np.histogram(background_predictions, bins=Config.BIN_EDGES, weights=background_weights,
                                          density=False)
        significance = 0.0
        threshold_index = np.digitize(threshold, Config.BIN_EDGES) - 1

        selected_signal = signal_hist[threshold_index:]
        selected_background = background_hist[threshold_index:]
        S = np.sum(selected_signal)
        B = np.sum(selected_background)

        if S > 0 and B > 0:
            significance = S / np.sqrt(S + B)
        else:
            significance = 0.0

        return significance


    def _signal_significance_plot(self, signal_predictions, background_predictions, signal_weights, background_weights):
        significances = list()
        thresholds = list()

        for threshold in Config.BIN_EDGES:
            significance = self._calculate_signal_significance(signal_predictions, background_predictions, signal_weights,
                                                               background_weights, threshold)
            significances.append(significance)
            thresholds.append(threshold)
        return significances, thresholds


    def signal_significance_plot(self, signal_predictions, background_predictions, signal_weights, background_weights):
        significances, thresholds = self._signal_significance_plot(signal_predictions,
                                                                   background_predictions,
                                                                   signal_weights,
                                                                   background_weights)
        max_significance = np.max(significances)
        max_index = np.argmax(significances)
        optimal_threshold = thresholds[max_index]

        figure, axis = plt.subplots(figsize=Config.PLOT_SIZE)
        axis.plot(Config.BIN_EDGES, significances, color='blue')
        axis.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Best threshold = {optimal_threshold:.3f}')
        axis.title('Signal Significance vs Threshold', fontsize=Config.FONT_SIZE)
        axis.xlabel('Classification Threshold', fontsize=Config.FONT_SIZE)
        axis.ylabel('Signal Significance', fontsize=Config.FONT_SIZE)
        axis.tick_params(axis='both', labelsize=Config.FONT_SIZE)
        axis.ticklabel_format(axis='both', style='sci', scilimits=Config.POWER_LIMITS)
        axis.legend(loc='best', fontsize=Config.FONT_SIZE, fancybox=False, edgecolor='black')

        self._current_figure = figure
        self._current_plot_name = 'significances'

        return figure
