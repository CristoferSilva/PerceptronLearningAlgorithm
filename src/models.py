from abc import ABC

import numpy as np
import scipy.stats as st
import abc
import src.optimizers as opt


class Model(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self._w = None

    @abc.abstractmethod
    def predict(self, X) -> np.ndarray:
        """Implement the predict method"""

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value


class LinearModel(Model, ABC):
    def __init__(self):
        super().__init__()

    def predict(self, X) -> np.ndarray:
        return X @ self.w
