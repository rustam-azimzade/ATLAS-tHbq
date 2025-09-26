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
import tensorflow as tf
import random
import os
from modules import Config, Events, Plots

DATA_DIR = '../01_src/01_data/02_json'
SAVE_PATH = '../03_results/03_neural_network/01_performance_plots'

WEIGHTS_SEED_NUMBER = 35
GLOBAL_SEED_NUMBER = 5
np.random.seed(GLOBAL_SEED_NUMBER)
tf.random.set_seed(GLOBAL_SEED_NUMBER)
random.seed(GLOBAL_SEED_NUMBER)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

def split_classes(predictions, output_true, weights):
    signal_mask = output_true == 1
    background_mask = output_true == 0

    signal_predictions = predictions[signal_mask]
    background_predictions = predictions[background_mask]

    signal_weights = weights[signal_mask]
    background_weights = weights[background_mask]

    return signal_predictions, background_predictions, signal_weights, background_weights


def main():
    events_handler = Events(data_dir=DATA_DIR)
    events_handler.load_data().normalize_variables().select_variables().label_data()
    input_data, output_data = events_handler.get_data()

    plotter = Plots()
    input_train, input_test, output_train, output_test = train_test_split(
        input_data, output_data,
        test_size=0.5,
        shuffle=True,
        random_state=GLOBAL_SEED_NUMBER,
        stratify=output_data
    )

    train_weights = events_handler.get_weights(input_train, equalize_classes=True)
    test_weights = events_handler.get_weights(input_train, equalize_classes=False)

    input_train.drop(columns=['weight', 'channel'], inplace=True)
    input_test.drop(columns=['weight', 'channel'], inplace=True)

    neural_network = load_model('../03_results/03_neural_network/02_pre-trained_model/tH(bb).hdf5')
    predictions_train = neural_network.predict(input_train).ravel()
    predictions_test = neural_network.predict(input_test).ravel()

    signal_train_predictions, background_train_predictions, signal_train_weights, background_train_weights = split_classes(predictions_train, output_train, train_weights)
    signal_test_predictions, background_test_predictions, signal_test_weights, background_test_weights = split_classes(predictions_test, output_test, test_weights)

    plotter.plot_roc_curve(predictions_test, output_test, test_weights)
    plotter.save_prediction_histogram(signal_test_predictions, background_test_predictions, signal_test_weights, background_test_weights)
    plotter.plot_significance_plot(signal_test_predictions, background_test_predictions, signal_test_weights, background_test_weights)


if __name__ == '__main__':
    main()