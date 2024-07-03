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


plt.style.use(['science', 'notebook', 'grid'])
matplotlib.rcParams.update({'font.size': 14})
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)


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

    columns_number = input_data.shape[1]

    input_train, input_test, output_train, output_test = train_test_split(input_data, output, test_size=0.3,
                                                                          shuffle=True)

    neural_network = Sequential([
        Dense(units=columns_number, input_dim=columns_number, activation='relu'),
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

    neural_network.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics='AUC')

    training_history = neural_network.fit(input_train, output_train, epochs=150, batch_size=256, validation_split=0.2)

    neural_network.save('tHbq_signal_classification.h5')

    check_overfitting(training_history)
    plot_roc_curve(neural_network, input_test, output_test)
    plot_histogram_of_predictions(neural_network, input_test, output_test)
    plot_separate_histogram_of_predictions(neural_network, input_test, output_test)


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


def check_overfitting(history):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    axis1.set_ylabel('Loss Function', fontsize=14)
    axis1.set_xlabel('Number of Epochs', fontsize=14)
    axis1.tick_params(axis='both', labelsize=10)
    axis1.plot(history.history['loss'], label='Train Data')
    axis1.plot(history.history['val_loss'], label='Validation Data')
    axis1.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_ylabel('AUC', fontsize=14)
    axis2.set_xlabel('Number of Epochs', fontsize=14)
    axis2.tick_params(axis='both', labelsize=10)
    axis2.plot(history.history['auc'], label='Train Data')
    axis2.plot(history.history['val_auc'], label='Validation Data')
    axis2.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


def plot_roc_curve(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()
    fpr, tpr, thresholds = roc_curve(outputs, predictions, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='best', fontsize=14, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


def plot_histogram_of_predictions(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()
    signal_predictions = predictions[outputs == 1]
    background_predictions = predictions[outputs == 0]

    common_bins = np.histogram_bin_edges(np.concatenate((signal_predictions, background_predictions)), bins='scott')
    signal_hist, _ = np.histogram(signal_predictions, bins=common_bins)
    background_hist, _ = np.histogram(background_predictions, bins=common_bins)
    separation_power = 0.5 * np.sum((signal_hist - background_hist) ** 2 / (signal_hist + background_hist + 1e-6))

    signal_bins = np.histogram_bin_edges(signal_predictions, bins='scott')
    background_bins = np.histogram_bin_edges(background_predictions, bins='scott')

    plt.figure()
    plt.title('Histogram of Neural Network Output', fontsize=14)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Number of events', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.hist(signal_predictions, bins=len(common_bins) * 10, alpha=0.5, label='Signal (p + p → t + H)',
             color='blue')
    plt.hist(background_predictions, bins=len(common_bins) * 10, alpha=0.7, hatch='//', histtype='step',
             label='Background', color='red')
    plt.legend(loc='upper center', fontsize=14, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


def plot_separate_histogram_of_predictions(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()

    variables = list(inputs_data.columns)

    for variable in variables:
        signal_predictions = inputs_data[outputs == 1][variable]

        high_signal_mask = (predictions > 0.95) & (outputs == 1)
        high_signal_predictions = inputs_data[high_signal_mask][variable]

        bins = np.histogram_bin_edges(signal_predictions, bins='scott')

        plt.figure()
        plt.title(f'Distribution of {variable} for Neural Network Signal Output Events', fontsize=14)
        plt.xlabel(f'Normalized Value of {variable}', fontsize=14)
        plt.ylabel('Number of events', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.hist(signal_predictions, bins=bins, alpha=0.5, label='All Signal Events', color='blue')
        plt.hist(high_signal_predictions, bins=bins, alpha=0.9, hatch='//', histtype='step', color='red',
                 label='High Signal Events (NN Output > 0.95)')
        plt.legend(loc='best', fontsize=14, fancybox=False, edgecolor='black')

        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
