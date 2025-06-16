from abc import ABC, abstractmethod

class BaseForecastModel(ABC):
    @abstractmethod
    def train_with_fold(self, folds, optimization=False):
        ...

    @abstractmethod
    def predict(self, pred_df, h=None):
        ...

    @abstractmethod
    def evaluate(self, actual_df, forecast_df):
        ...

    @abstractmethod
    def save(self, *args, **kwargs):
        ...

    @abstractmethod
    def create_folds(self, df, n_splits, test_size):
        ...

    @abstractmethod
    def optimize(self, df, *args, **kwargs):
        ...