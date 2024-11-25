import keras.optimizers
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from config import Config


FONT_SIZE = 14
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((0, 0))


class EvaluateWithoutDropout(Callback):
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data

    def on_epoch_end(self, epoch, logs=None):
        train_loss, train_auc = self.model.evaluate(self.train_data[0], self.train_data[1], verbose=0)
        logs['loss'] = train_loss
        logs['auc'] = train_auc


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.useoffset': False,
        'axes.formatter.offset_threshold': 1
    })


def define_model(input_neurons):
    model = Sequential([
        Dense(units=input_neurons, input_dim=input_neurons, activation='relu', kernel_initializer=HeNormal()),
        BatchNormalization(),
        Dropout(0.5),
        Dense(units=128, activation='relu', kernel_initializer=HeNormal()),
        BatchNormalization(),
        Dropout(0.5),
        Dense(units=64, activation='relu', kernel_initializer=HeNormal()),
        BatchNormalization(),
        Dropout(0.5),
        Dense(units=1, activation='sigmoid', kernel_initializer=HeNormal())
    ])

    my_optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics='AUC')
    return model


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


def exponential_decay(epoch, patience_epoch, initial_learning_rate, final_learning_rate, decay_factor):
    learning_rate = initial_learning_rate

    if epoch >= patience_epoch:
        k = -np.log(final_learning_rate / initial_learning_rate) / decay_factor
        custom_learning_rate = initial_learning_rate * np.exp(-k * (epoch - patience_epoch))
        learning_rate = max(custom_learning_rate, final_learning_rate)

    return learning_rate


def save_history(history):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axis1 = axes[0]
    axis1.set_ylabel('Binary Crossentropy', fontsize=FONT_SIZE)
    axis1.set_xlabel('Number of Epochs', fontsize=FONT_SIZE)
    axis1.tick_params(axis='both', labelsize=FONT_SIZE)
    axis1.plot(history.history['loss'], label='Train Data')
    axis1.plot(history.history['val_loss'], label='Validation Data')
    axis1.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_ylabel('AUC', fontsize=FONT_SIZE)
    axis2.set_xlabel('Number of Epochs', fontsize=FONT_SIZE)
    axis2.tick_params(axis='both', labelsize=FONT_SIZE)
    axis2.plot(history.history['auc'], label='Train Data')
    axis2.plot(history.history['val_auc'], label='Validation Data')

    axis2.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    plt.savefig('../03_results/02_neural_network/training_history.png', dpi=300)
    plt.close()


def save_roc_curve(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()
    fpr, tpr, thresholds = roc_curve(outputs, predictions, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic', fontsize=FONT_SIZE)
    plt.ylabel('True Positive Rate', fontsize=FONT_SIZE)
    plt.xlabel('False Positive Rate', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    plt.savefig('../03_results/02_neural_network/roc_curve.png', dpi=300)
    plt.close()


def save_histogram_of_predictions(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()
    signal_predictions = predictions[outputs == 1]
    background_predictions = predictions[outputs == 0]

    bins = np.linspace(0, 1, 20)

    signal_hist, _ = np.histogram(signal_predictions, bins=bins, density=True)
    background_hist, _ = np.histogram(background_predictions, bins=bins, density=True)

    signal_hist /= np.sum(signal_hist)
    background_hist /= np.sum(background_hist)
    separation_power = 0
    for i in range(1, len(signal_hist) - 1):
        if signal_hist[i] == 0 and background_hist[i] == 0:
            continue
        separation_power += (signal_hist[i] - background_hist[i]) ** 2 / (signal_hist[i] + background_hist[i] + 1e-10)
    separation_power *= 0.5

    signal_significance = 0
    for i in range(1, len(signal_hist) - 1):
        if signal_hist[i] == 0 and background_hist[i] == 0:
            continue
        signal_significance += (signal_hist[i] / np.sqrt(signal_hist[i] + background_hist[i] + 1e-10))

    plt.figure()
    plt.title('Histogram of Neural Network Output', fontsize=FONT_SIZE)
    plt.xlabel('Predicted Probability', fontsize=FONT_SIZE)
    plt.ylabel('Number of events', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.hist(signal_predictions, bins=bins, alpha=0.9, hatch='//', histtype='step',
             label='Signal (p + p → t + H)', color='red')
    plt.hist(background_predictions, bins=bins, alpha=0.4, label='Background', color='blue')
    plt.legend(loc='upper center', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')
    plt.annotate(f'Separation Power: {separation_power * 100:.2f}%\nSignal Significance: {signal_significance * 100:.2f}%',
                 xy=(0.28, 0.80), xycoords='axes fraction', fontsize=FONT_SIZE, verticalalignment='top',
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1))
    plt.savefig('../03_results/02_neural_network/prediction.png', dpi=300)
    plt.close()


def save_separate_histogram_of_predictions(model, inputs_data, outputs):
    predictions = model.predict(inputs_data).ravel()

    variables = list(inputs_data.columns)

    for variable in variables:
        signal_predictions = inputs_data[outputs == 1][variable]

        high_signal_mask = (predictions > 0.95) & (outputs == 1)
        high_signal_predictions = inputs_data[high_signal_mask][variable]

        bins = np.histogram_bin_edges(signal_predictions, bins='scott')

        plt.figure()
        plt.title(f'Distribution of {variable} for Neural Network Signal Output Events', fontsize=FONT_SIZE)
        plt.xlabel(f'Normalized Value of {variable}', fontsize=FONT_SIZE)
        plt.ylabel('Number of events', fontsize=FONT_SIZE)
        plt.tick_params(axis='both', labelsize=FONT_SIZE)
        plt.hist(signal_predictions, bins=bins, alpha=0.4, label='All Signal Events', color='blue')
        plt.hist(high_signal_predictions, bins=bins, alpha=0.9, hatch='//', histtype='step', color='red',
                 label='High Signal Events (NN Output > 0.95)')
        plt.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

        plt.show()
        plt.close()


def main():
    signal_data = pd.read_json('../data/json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    background_data = pd.read_json('../data/json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')

    background_data_tt = pd.read_json('../data/json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')
    background_data_ttbb = pd.read_json('../data/json/MiniNtuple_ttbb_SM_300K_(aTTreett;1).json')
    background_data_ttH = pd.read_json('../data/json/MiniNtuple_ttH_SM_100K_(aTTreetth;1).json')
    background_data_tzbq = pd.read_json('../data/json/MiniNtuple_tzbq_SM_100K_(aTTreethbq;1).json')

    #background_data = pd.concat([background_data_tt, background_data_ttbb, background_data_ttH, background_data_tzbq],
    #                            axis=0)

    signal_data = signal_data.drop(columns=['N_b'])
    background_data = background_data.drop(columns=['N_b'])

    for branch_name in Config.VARIABLES_DESCRIPTION:
        signal_data[branch_name], max_value, min_value = normalize(signal_data[branch_name])
        background_data[branch_name], _, _ = normalize(background_data[branch_name], max_value, min_value)

    # Label data
    signal_data['signal'] = 1
    background_data['signal'] = 0

    # Prepare data
    total_data = pd.concat([signal_data, background_data])
    total_data = total_data.sample(frac=1).reset_index(drop=True)
    total_data.index = range(1, len(total_data) + 1)
    input_data = total_data.drop(columns=['signal'])
    output = pd.Series(total_data['signal'])

    input_train, input_test, output_train, output_test = train_test_split(input_data, output, test_size=0.3,
                                                                        shuffle=True)

    columns_number = input_data.shape[1]
    neural_network = define_model(input_neurons=columns_number)

    learning_rate_scheduler = LearningRateScheduler(
        lambda epoch: exponential_decay(epoch, patience_epoch=200, initial_learning_rate=0.001,
                                        final_learning_rate=0.0001, decay_factor=80)
    )
    tensor_board = TensorBoard(log_dir='03_results/tb_logs', histogram_freq=1, write_images=True)
    evaluate_without_dropout = EvaluateWithoutDropout(train_data=(input_train, output_train))
    early_stop_callback = EarlyStopping(monitor='val_auc', mode='max', patience=20, restore_best_weights=True, verbose=1)
    callbacks = [learning_rate_scheduler, evaluate_without_dropout, early_stop_callback]

    training_history = neural_network.fit(input_train, output_train, epochs=5000, batch_size=1024, verbose=1,
                                          callbacks=callbacks, validation_split=0.2, shuffle=True)
    neural_network.save('../03_results/02_neural_network/tH(bb)_signal_classification.hdf5')

    save_history(training_history)
    save_roc_curve(neural_network, input_test, output_test)
    save_histogram_of_predictions(neural_network, input_test, output_test)
    #save_separate_histogram_of_predictions(neural_network, input_test, output_test)


if __name__ == '__main__':
    set_plot_style()
    main()
