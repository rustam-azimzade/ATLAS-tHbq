import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Iterable
from .config import Config

logging.basicConfig(
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    level=logging.INFO
)
logging.getLogger('fontTools').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class Events:
    def __init__(self, data_dir=Config.DATA_DIR):
        self.data_dir = Path(data_dir)
        self.channels_events = dict()

    def load_data(self):
        # Load all datasets
        for channel, filename in Config.EVENT_FILES.items():
            file_path = self.data_dir / filename
            try:
                self.channels_events[channel] = pd.read_json(file_path)
                logger.info(f"Channel {channel} is loaded from {file_path}")
            except Exception as e:
                logger.error(f"Channel loading error {file_path}: {e}", exc_info=True)
        return self


    def normalize(self, data: pd.Series, max_value=None, min_value=None):
        if not isinstance(data, pd.DataFrame):
            data = data.to_frame()

        if max_value is None or min_value is None:
            max_value = data.max()
            min_value = data.min()

        data = (data - min_value) / (max_value - min_value)
        return data, max_value, min_value


    def normalize_variables(self):
        signal_events = self.channels_events[Config.SIGNAL_CHANNEL_NAME]
        for variable_name in signal_events.columns:
            # Normalization features for signal channel
            signal_events[variable_name], max_value, min_value = self.normalize(signal_events[variable_name])

            # Apply the same normalization to background channels
            for channel_name, channel_data in self.channels_events.items():
                if channel_name != Config.SIGNAL_CHANNEL_NAME:
                    channel_data[variable_name], _, _ = self.normalize(
                        channel_data[variable_name],
                        max_value, min_value
                    )
        logger.info("Variable normalization completed.")
        return self


    def select_variables(self):
        selected_variables = list(Config.VARIABLES_DESCRIPTION.keys())

        for channel_name in self.channels_events:
            self.channels_events[channel_name] = self.channels_events[channel_name][selected_variables].copy()
        logger.info(f"Variables selected: {selected_variables}")
        return self


    def label_data(self):
        for channel_name, channel_events in self.channels_events.items():
            channel_events['channel'] = channel_name
            channel_events['signal'] = 1 if channel_name == Config.SIGNAL_CHANNEL_NAME else 0
        logger.info("Data is marked.")
        return self


    def get_signal_events(self):
        signal_events = self.channels_events.get(Config.SIGNAL_CHANNEL_NAME, pd.DataFrame()).copy()
        return signal_events


    def get_background_events(self):
        background_channel_names = [
            channel_name for channel_name in self.channels_events
            if channel_name != Config.SIGNAL_CHANNEL_NAME
        ]
        background_events = pd.concat(
            [self.channels_events[channel_name] for channel_name in background_channel_names],
            ignore_index=True
        )
        return background_events


    def get_data(self):
        total_events = pd.concat(self.channels_events.values(), ignore_index=True)
        return total_events


    ## Тут реализовать получение весов чтобы потом удалить названия колонок каналов в следующей функции
    def get_weight(self, channel_name, events_number):
        weight = (Config.LUMINOSITY * Config.CROSS_SECTIONS[channel_name] / Config.GENERATED_EVENTS_NUMBER[channel_name]) * (Config.PRESELECTION_EVENTS_NUMBER[channel_name] / events_number)
        logger.info(f"Channel {channel_name} — Weight {weight}")
        return weight


    def get_weights(self, data, equalize_classes=False):
        unique_channels = data['channel'].value_counts()
        for channel, num_events in unique_channels.items():
            event_weight = self.get_weight(channel_name=channel, events_number=num_events)
            data.loc[data['channel'] == channel, 'weight'] = event_weight

        if equalize_classes:
            # Считаем взвешанное число фоновых событий
            background_total_weighted_events = 0.0

            for channel, num_events in unique_channels.items():
                if channel != Config.SIGNAL_CHANNEL_NAME:
                    channel_weight = data[data['channel'] == channel]['weight'].iloc[0]
                    weighted_channel_events = channel_weight * num_events
                    background_total_weighted_events += weighted_channel_events

            # Присваиваем tHbq такой вес чтобы сигнальных событий было столько же как общего фона
            num_signal_events = unique_channels.get(Config.SIGNAL_CHANNEL_NAME, 0)
            signal_weight = background_total_weighted_events / num_signal_events
            data.loc[data['channel'] == Config.SIGNAL_CHANNEL_NAME, 'weight'] = signal_weight

        # Подготовка данных
        weights = pd.Series(data['weight']).copy()
        data.drop(columns=['weight'], inplace=True)

        return weights


    def select_variable(self, variable_name: str):
        for channel_name in self.channels_events:
            self.channels_events[channel_name] = self.channels_events[channel_name][[variable_name]].copy()
        logger.info(f"Only the variable {variable_name} is left")
        return self


    @staticmethod
    def filter_variables(variables: Dict[str, str], excluded: Iterable[str]) -> Dict[str, str]:
        excluded = set(excluded)

        selected = {
            key: value
            for key, value in variables.items()
            if key not in excluded
        }
        return selected


    def get_variable_names(self):
        first_channel = next(iter(self.channels_events.values()))
        variable_names = list(first_channel.columns)

        for technical_column_name in ['channel', 'signal']:
            if technical_column_name in variable_names:
                variable_names.remove(technical_column_name)

        logger.info(f"Current variables: {variable_names}")
        return variable_names


    def get_splited_data(self):
        total_events = self.get_data()
        input_data = total_events.copy()

        output_data = total_events['signal'].copy()

        return input_data, output_data


    @staticmethod
    def get_score_after_permutation(
            model,
            input_data: pd.DataFrame,
            target: pd.Series,
            feature_name: str
    ) -> float:
        input_data_permuted = input_data.copy()
        input_data_permuted[feature_name] = np.random.permutation(
            input_data_permuted[feature_name].values
        )
        permuted_score = model.score(input_data_permuted, target)
        return permuted_score


    @classmethod
    def calculate_feature_importance(
            cls,
            model,
            input_data: pd.DataFrame,
            target: pd.Series,
            feature_name: str,
            n_repeats: int = Config.PERMUTATION_REPEATS
    ) -> np.ndarray:
        baseline_score = model.score(input_data, target)
        feature_importances = np.empty(n_repeats)

        for i in range(n_repeats):
            permuted_score = cls.get_score_after_permutation(model, input_data, target, feature_name)
            feature_importances[i] = baseline_score - permuted_score

        return feature_importances


    @classmethod
    def permutation_importance(
            cls,
            model,
            input_data: pd.DataFrame,
            target: pd.Series,
            n_repeats: int = Config.PERMUTATION_REPEATS
    ) -> pd.DataFrame:
        results = list()

        for feature_name in input_data.columns:
            importances = cls.calculate_feature_importance(
                model, input_data, target, feature_name, n_repeats=n_repeats
            )
            results[feature_name] = {
                "importances_mean": np.mean(importances),
                "importances_std": np.std(importances),
                "importances": importances
            }

            return pd.DataFrame(results).T.sort_values("importances_mean", ascending=False)
