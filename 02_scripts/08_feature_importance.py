import numpy as np
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from modules import Config, Events
import tensorflow as tf
import random
import os
from sklearn.inspection import permutation_importance

DATA_DIR = '../01_src/01_data/02_json'
WEIGHTS_SEED_NUMBER = 35
GLOBAL_SEED_NUMBER = 5

np.random.seed(GLOBAL_SEED_NUMBER)
tf.random.set_seed(GLOBAL_SEED_NUMBER)
random.seed(GLOBAL_SEED_NUMBER)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

PLOTS_SAVE_PATH = '../03_results/03_neural_network/01_performance_plots'


def plot_feature_importances(feature_importances, feature_names):
    feature_importances_mean = feature_importances.importances_mean
    feature_importances_std = feature_importances.importances_std

    # Сортировка по важности
    indices = np.argsort(feature_importances_mean)

    fig, ax = plt.subplots(figsize=(8, len(feature_names) * 0.4 + 1))
    ax.barh(range(len(indices)), feature_importances_mean[indices], xerr=feature_importances_std[indices], align="center")
    #plt.barh(features, feature_importances_mean, xerr=feature_importances_std)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(np.array(feature_names)[indices])
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_title("Permutation Feature Importances")
    plt.tight_layout()
    plt.show()


def main():
    events_handler = Events(data_dir=DATA_DIR)
    events_handler.load_data().process_data()
    input_data, output_data = events_handler.get_data()

    input_train, input_test, output_train, output_test = train_test_split(
        input_data, output_data,
        test_size=0.5,
        shuffle=True,
        random_state=GLOBAL_SEED_NUMBER,
        stratify=output_data
    )

    neural_network = load_model('../03_results/03_neural_network/02_pre-trained_model/tH(bb).hdf5')
    wrapped_model = KerasClassifier(model=neural_network, optimizer=None, loss=None)

    feature_importances = permutation_importance(wrapped_model, input_test, output_test,
                                                 n_repeats=30,
                                                 random_state=GLOBAL_SEED_NUMBER,
                                                 scoring='roc_auc')

    features = input_test.feature_names
    plot_feature_importances(feature_importances, features)


if __name__ == '__main__':
    main()
