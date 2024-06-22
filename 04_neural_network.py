import keras.optimizers
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from config import Config


def main():
    signal_data = pd.read_json('data/json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    background_data = pd.read_json('data/json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')

    for branch_name in Config.VARIABLES_DESCRIPTION:
        signal_data[branch_name], max_value, min_value = normalize(signal_data[branch_name])
        background_data[branch_name], _, _ = normalize(background_data[branch_name], max_value, min_value)

    signal_data['signal'] = 1
    background_data['signal'] = 0

    total_data = pd.concat([signal_data, background_data])
    total_data = total_data.sample(frac=1).reset_index(drop=True)
    total_data.index = range(1, len(total_data) + 1)

    input_data = total_data.drop(columns=['signal'])
    output = pd.Series(total_data['signal'])

    input_train, input_test, output_train, output_test = train_test_split(input_data, output,
                                                                          test_size=0.3,
                                                                          shuffle=True)

    neural_network = Sequential([
        Dense(units=24, input_dim=24, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(units=128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(units=64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(units=1, activation='sigmoid')
    ])

    my_optimizer = keras.optimizers.Adam(learning_rate=0.001)

    neural_network.compile(optimizer=my_optimizer, loss='binary_crossentropy')

    neural_network.fit(input_train, output_train, epochs=150, batch_size=256, verbose=1)

    neural_network.save('tHbq_signal_classification.h5')

    plot_roc_curve(neural_network, input_test, output_test)
    plot_histogram_of_predictions(neural_network, input_test, output_test)


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


def plot_roc_curve(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()
    fpr, tpr, thresholds = roc_curve(outputs, predictions, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.style.use(['science', 'notebook', 'grid'])
    matplotlib.rcParams.update({'font.size': 14})
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)

    plt.figure()
    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='best', fontsize=14, fancybox=False, edgecolor='black')
    plt.show()


def plot_histogram_of_predictions(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()
    signal_predictions = predictions[outputs == 1]
    signal_predictions_bins = np.histogram_bin_edges(signal_predictions, bins='scott')
    background_predictions = predictions[outputs == 0]
    background_predictions_bins = np.histogram_bin_edges(signal_predictions, bins='scott')

    plt.style.use(['science', 'notebook', 'grid'])
    matplotlib.rcParams.update({'font.size': 14})
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)

    plt.figure()
    plt.title('Histogram of Neural Network Output', fontsize=14)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.hist(signal_predictions, bins=signal_predictions_bins, alpha=0.5, label='Signal (tHbq)', color='green')
    plt.hist(background_predictions, bins=background_predictions_bins, alpha=0.8, hatch='//', histtype='step', label='Background', color='red')
    plt.legend(loc='upper center', fontsize=14, fancybox=False, edgecolor='black')
    plt.show()


if __name__ == '__main__':
    main()
