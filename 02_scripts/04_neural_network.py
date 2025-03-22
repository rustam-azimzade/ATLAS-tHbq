import numpy as np
from jax.example_libraries.stax import randn
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from config import Config
import tensorflow as tf
import random
import os
import optuna

WEIGHTS_SEED_NUMBER = 35
GLOBAL_SEED_NUMBER = 5
FONT_SIZE = 14

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

best_neural_network = None
best_auc_score = 0.0
best_neural_network_training_history = None

PLOTS_SAVE_PATH = '../03_results/03_neural_network/01_performance_plots'

class EvaluateWithoutDropout(Callback):
    def __init__(self, train_data, sample_weight=None):
        super().__init__()
        self.train_data = train_data
        self.sample_weight = sample_weight

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(
            self.train_data[0],
            self.train_data[1],
            sample_weight=self.sample_weight,
            verbose=0
        )

        for name, value in zip(self.model.metrics_names, results):
            logs[name] = value


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.useoffset': False,
        'axes.formatter.offset_threshold': 1
    })


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

    return total_events


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


def save_history(history):
    plt.figure()

    plt.ylabel('Weighted Binary Crossentropy', fontsize=FONT_SIZE)
    plt.xlabel('Number of Epochs', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.plot(history.history['weighted_binary_crossentropy'], label='Train Data')
    plt.plot(history.history['val_weighted_binary_crossentropy'], label='Validation Data')
    plt.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    plt.savefig(f'{PLOTS_SAVE_PATH}/01_png/training_history.png', dpi=300)
    plt.savefig(f'{PLOTS_SAVE_PATH}/02_pdf/training_history.pdf')
    plt.close()


def save_roc_curve(model, inputs_data, outputs, weights):
    predictions = model.predict(inputs_data).ravel()
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
    plt.savefig(f'{PLOTS_SAVE_PATH}/02_pdf/roc_curve.pdf')
    plt.close()


def save_histogram_of_predictions(model, inputs_data, outputs, weights=None, significance_weights=None):
    bins = np.linspace(0, 1, 30)

    predictions = model.predict(inputs_data).ravel()

    signal_mask = outputs == 1
    background_mask = outputs == 0

    signal_predictions = predictions[signal_mask]
    background_predictions = predictions[background_mask]

    signal_weights = weights[signal_mask]
    background_weights = weights[background_mask]
    signal_hist, _ = np.histogram(signal_predictions, bins=bins, weights=signal_weights, density=True)
    background_hist, _ = np.histogram(background_predictions, bins=bins, weights=background_weights, density=True)
    signal_hist /= np.sum(signal_hist)
    background_hist /= np.sum(background_hist)

    separation_power = 0
    for i in range(1, len(signal_hist) - 1):
        if signal_hist[i] == 0 and background_hist[i] == 0:
            continue
        separation_power += (signal_hist[i] - background_hist[i]) ** 2 / (signal_hist[i] + background_hist[i])
    separation_power *= 0.5

    plt.figure()
    plt.title('Histogram of Neural Network Output', fontsize=FONT_SIZE)
    plt.xlabel('Predicted Probability', fontsize=FONT_SIZE)
    plt.ylabel('Number of events', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.hist(signal_predictions, bins=bins, alpha=0.9, hatch='//', histtype='step', label='Signal (p + p → t + H)',
             color='red', weights=signal_weights)
    plt.hist(background_predictions, bins=bins, alpha=0.4, label='Background', color='blue', weights=background_weights)
    plt.legend(loc='upper center', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    signal_weights = significance_weights[signal_mask]
    background_weights = significance_weights[background_mask]
    signal_hist, _ = np.histogram(signal_predictions, bins=bins, weights=signal_weights, density=True)
    background_hist, _ = np.histogram(background_predictions, bins=bins, weights=background_weights, density=True)
    signal_hist /= np.sum(signal_hist)
    background_hist /= np.sum(background_hist)

    signal_significance = 0
    for i in range(1, len(signal_hist) - 1):
        if signal_hist[i] == 0 and background_hist[i] == 0:
            continue
        signal_significance += (signal_hist[i] / np.sqrt(signal_hist[i] + background_hist[i]))

    plt.annotate(f'Separation Power: {separation_power * 100:.2f}%\nSignal Significance: {signal_significance * 100:.2f}%',
                 xy=(0.28, 0.80), xycoords='axes fraction', fontsize=FONT_SIZE, verticalalignment='top',
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1))

    plt.savefig(f'{PLOTS_SAVE_PATH}/01_png/prediction.png', dpi=300)
    plt.savefig(f'{PLOTS_SAVE_PATH}/02_pdf/prediction.pdf')
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


def define_model(input_neurons, trial):
    # Hyperparameters to optimize
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 5, step=1)
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4])
    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'SGD', 'RMSprop', 'Nadam'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    activation_l1 = trial.suggest_categorical('activation_l1', ['relu', 'tanh', 'swish'])
    dropout_l1 = trial.suggest_float('dropout_l1', 0.0, 0.5, step=0.1)
    #l1_reg = trial.suggest_loguniform('l1_reg', 1e-7, 1e-3)
    #l2_reg = trial.suggest_loguniform('l2_reg', 1e-7, 1e-3)

    # Define model
    model = Sequential()
    model.add(Dense(units=input_neurons, input_dim=input_neurons, activation=activation_l1, kernel_initializer=HeNormal(seed=WEIGHTS_SEED_NUMBER)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_l1))

    for i in range(n_hidden_layers):
        n_neurons = trial.suggest_int(f'n_neurons_l{i + 2}', 32, 512, step=5)
        dropout = trial.suggest_float(f'dropout_l{i + 2}', 0.0, 0.5, step=0.1)
        activation = trial.suggest_categorical(f'activation_l{i + 2}', ['relu', 'tanh', 'swish'])
        model.add(Dense(units=n_neurons, activation=activation, kernel_initializer=HeNormal(seed=WEIGHTS_SEED_NUMBER)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=HeNormal(seed=WEIGHTS_SEED_NUMBER)))

    optimizer = Adam(learning_rate=learning_rate)
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_crossentropy'],
        weighted_metrics=['binary_crossentropy']
    )
    return model, batch_size


def objective(trial, input_train, input_test, output_train, output_test, weights_train, weights_test):
    global best_neural_network, best_auc_score, best_neural_network_training_history

    columns_number = input_train.shape[1]
    neural_network, batch_size = define_model(input_neurons=columns_number, trial=trial)

    evaluate_without_dropout = EvaluateWithoutDropout(
        train_data=(input_train, output_train),
        sample_weight=weights_train
    )
    early_stop_callback = EarlyStopping(
        monitor='val_weighted_binary_crossentropy',
        mode='min',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    callbacks = [evaluate_without_dropout, early_stop_callback]

    training_history = neural_network.fit(
        input_train, output_train,
        epochs=5000,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        sample_weight=weights_train,
        validation_split=0.2,
        shuffle=True,
        workers=-1,  # Использует все доступные ядра для загрузки данных
        use_multiprocessing=True  # Включает многозадачность для обработки данных
    )

    output_predicted = neural_network.predict(input_test).ravel()
    fpr, tpr, thresholds = roc_curve(output_test, output_predicted, sample_weight=weights_test, drop_intermediate=False)
    auc_score = auc(fpr, tpr)

    if auc_score > best_auc_score:
        best_auc_score = auc_score
        best_neural_network = neural_network
        best_neural_network_training_history = training_history

    return auc_score


def run_optimization(input_train, input_test, output_train, output_test, weights_train, weights_test):
    study = optuna.create_study(
        study_name='Hyperparameter_optimization',
        direction='maximize',
        storage='sqlite:///../03_results/03_neural_network/optimization.db',
        load_if_exists=True
    )
    study.optimize(
        lambda trial: objective(trial, input_train, input_test, output_train, output_test, weights_train, weights_test),
        n_trials=1,
        n_jobs=-1
    )
    return study


def main():
    global best_neural_network, best_auc_score, best_neural_network_training_history

    total_events = load_data()

    input_data = total_events.drop(columns=['signal', 'weight', 'significance_weight'])
    output_data = pd.Series(total_events['signal'])
    events_weights = total_events['weight']
    significance_weights = total_events['significance_weight']

    input_train, input_test, output_train, output_test, weights_train, weights_test, significance_weights_train, significance_weights_test = train_test_split(
        input_data, output_data, events_weights, significance_weights,
        test_size=0.3,
        shuffle=True,
        random_state=GLOBAL_SEED_NUMBER,
        stratify=output_data
    )

    optimization_history = run_optimization(input_train, input_test, output_train, output_test, weights_train, weights_test)
    optimization_history.trials_dataframe().to_json('../03_results/03_neural_network/optuna_study_results.json',
                                                    orient='records',
                                                    lines=True
                                                    )
    print('Best trial:')
    print(f'  Value: {optimization_history.best_trial.value}')
    print('  Params: ')
    for key, value in optimization_history.best_trial.params.items():
        print(f'    {key}: {value}')

    best_neural_network.save('../03_results/03_neural_network/02_pre-trained_model/tH(bb).hdf5')
    save_history(best_neural_network_training_history)
    save_roc_curve(best_neural_network, input_test, output_test, weights_test)
    save_histogram_of_predictions(best_neural_network, input_test, output_test, weights_test, significance_weights_test)


if __name__ == '__main__':
    set_plot_style()
    main()
