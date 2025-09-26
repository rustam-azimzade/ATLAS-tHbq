import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import load_model
from .config import Config


class NeuralNetwork:
    def __init__(self, input_dim, learning_rate=1e-4):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.training_history = None

    def _build_model(self):
        model = Sequential([
            # INPUT LAYER
            Dense(units=self.input_dim, activation='relu', kernel_initializer=HeNormal(seed=Config.WEIGHTS_SEED_NUMBER)),
            BatchNormalization(),
            Dropout(0.5),
            # FIRST HIDDEN LAYER
            Dense(units=128, activation='relu', kernel_initializer=HeNormal(seed=Config.WEIGHTS_SEED_NUMBER)),
            BatchNormalization(),
            Dropout(0.5),
            # SECOND HIDDEN LAYER
            Dense(units=64, activation='relu', kernel_initializer=HeNormal(seed=Config.WEIGHTS_SEED_NUMBER)),
            BatchNormalization(),
            Dropout(0.5),
            # OUTPUT LAYER
            Dense(units=1, activation='sigmoid', kernel_initializer=HeNormal(seed=Config.WEIGHTS_SEED_NUMBER))
        ])

        my_optimizer = Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=my_optimizer,
            loss='binary_crossentropy',
            metrics=['binary_crossentropy'],
            weighted_metrics=['binary_crossentropy']
        )
        return model

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

    def get_callbacks(self, train_data, output_data, train_weights, patience=20):
        evaluate_without_dropout = self.EvaluateWithoutDropout(
            train_data=(train_data, output_data),
            sample_weight=train_weights
        )
        early_stopping = EarlyStopping(
            monitor='val_weighted_binary_crossentropy',
            mode='min',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        return [evaluate_without_dropout, early_stopping]

    def train(self, input_data, output_data, sample_weight, batch_size=128, epochs=5000, validation_split=0.2):
        callbacks = self.get_callbacks(input_data, output_data, sample_weight)
        self.training_history = self.model.fit(
            input_data, output_data,
            sample_weight=sample_weight,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def get_history(self):
        return self.training_history

    def save_history(self, path):
        if self.training_history is None:
            raise RuntimeError("Training history is not available. You must call train() before saving history.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        history_df = pd.DataFrame(self.training_history.history)
        history_df.to_csv(path, index=False)

    @staticmethod
    def load_model(path):
        return load_model(path)
