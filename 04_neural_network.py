import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import RocCurveDisplay, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    matplotlib.rcParams.update({'font.size': 14})
    pd.set_option('display.max_columns', 10)

    signal_data_frame = pd.read_json('data/json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    signal_data_frame['target'] = 1

    background_data_frame = pd.read_json('data/json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')
    background_data_frame['target'] = 0

    inputs = pd.concat([signal_data_frame, background_data_frame])
    inputs = inputs.sample(frac=1).reset_index(drop=True)
    inputs.index = range(1, len(inputs) + 1)




    data = load_breast_cancer()

    input = pd.DataFrame(data['data'], columns=data['feature_names'])
    output = pd.Series(data['target'])

    np.random.seed(5)
    features = np.random.randint(input.shape[1], size=2)

    input_train, input_test, output_train, output_test = train_test_split(input.iloc[:, features], output, test_size=9, random_state=4)

    model = Sequential([
        Dense(22, input_dim=22, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])

    model.fit(input_train, output_train, epochs=50, batch_size=256, validation_split=0.2, verbose=1)

    # Сохранение модели
    model.save('atlas_classification_model.h5')

    RocCurveDisplay.from_estimator(model, input_test, output_test)
    plt.show()


def normalize(data, max_value=None, min_value=None):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()

    if max_value is None and min_value is None:
        max_value = data.max()
        min_value = data.min()

    data = (data - min_value) / (max_value - min_value)
    return data, max_value, min_value


if __name__ == '__main__':
    main()
