from abc import ABC

import numpy as np
import abc
import src.optimizers as opt
import src.models as models
import sklearn.metrics
import math
import src.stop_criteria as stop
from src.preprocessing import Preprocessing


class Algorithm(abc.ABC):
    def __init__(self, optimizer_strategy: opt.OptimizerStrategy, model: models.Model) -> None:
        self.algorithm_observers = []
        self.optimizer_strategy = optimizer_strategy
        self.model = model

    def add(self, observer):
        if observer not in self.algorithm_observers:
            self.algorithm_observers.append(observer)
        else:
            print('Failed to add: {}'.format(observer))

    def remove(self, observer):
        try:
            self.algorithm_observers.remove(observer)
        except ValueError:
            print('Failed to remove: {}'.format(observer))

    def notify_iteration(self):
        [o.notify_iteration(self) for o in self.algorithm_observers]

    def notify_started(self):
        [o.notify_started(self) for o in self.algorithm_observers]

    def notify_finished(self):
        [o.notify_finished(self) for o in self.algorithm_observers]

    @abc.abstractmethod
    def fit(self, X, Y, stop_criteria):
        """Implement the fit method"""

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value

    @property
    def errors(self):
        return self._errors

    @errors.setter
    def errors(self, value):
        self._errors = value

    @property
    def rmse(self):
        return self._rmse

    @rmse.setter
    def rmse(self, value):
        self._rmse = value


class PLA(Algorithm, ABC):
    def __init__(self, optimizer_strategy: opt.OptimizerStrategy, model: models.Model) -> None:
        super().__init__(optimizer_strategy, model)
    def fit(self, X, Y, stop_criteria):
        self.iteration = 0
        run_loop = True
        self.process_data(X)

        while run_loop:
            self.calculate_error(Y)
            self.notify_iteration()
            self.iteration += 1
            self.optimizer_strategy.update_model(self.X, Y, self.model)
            run_loop = not stop_criteria.isFinished(self)

        self.notify_finished()

    def calculate_error(self, Y):
        self.Y_hat = self.model.predict(self.X)
        self.errors = self.Y_hat - Y
        self.rmse = 1.0 / len(self.X) * np.square(np.linalg.norm(self.errors))

    def process_data(self, X):
        self.X = Preprocessing.build_design_matrix(X)
        self.model.w = Preprocessing.initilize_weights(X)
