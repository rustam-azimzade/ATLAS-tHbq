import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from modules import Events, NeuralNetwork

import tensorflow as tf
GLOBAL_SEED_NUMBER = 5

np.random.seed(GLOBAL_SEED_NUMBER)
tf.random.set_seed(GLOBAL_SEED_NUMBER)
random.seed(GLOBAL_SEED_NUMBER)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

DATA_DIR = '../01_src/01_data/02_json'
PLOTS_SAVE_PATH = '../03_results/03_neural_network/01_performance_plots'
MODEL_SAVE_PATH = '../03_results/03_neural_network/02_pre-trained_model/tH(bb).hdf5'
HISTORY_SAVE_PATH = '../03_results/03_neural_network/training_history.csv'


def main():
    events_handler = Events(data_dir=DATA_DIR)
    events_handler.load_data().normalize_variables().select_variables().label_data()
    input_data, output_data = events_handler.get_splited_data()

    input_train, input_test, output_train, output_test = train_test_split(
        input_data, output_data,
        test_size=0.5,
        shuffle=True,
        random_state=GLOBAL_SEED_NUMBER,
        stratify=output_data
    )

    train_weights = events_handler.get_weights(data=input_train, equalize_classes=True)

    input_train.drop(columns=['signal', 'channel'], inplace=True)
    input_test.drop(columns=['signal','channel'], inplace=True)

    num_input_neurons = input_train.shape[1]
    neural_network = NeuralNetwork(input_dim=num_input_neurons)
    neural_network.train(input_data=input_train, output_data=output_train, sample_weight=train_weights)

    neural_network.save_model(path=MODEL_SAVE_PATH)
    neural_network.save_history(path=HISTORY_SAVE_PATH)


if __name__ == '__main__':
    main()
