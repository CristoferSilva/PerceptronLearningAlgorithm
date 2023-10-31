from tokenize import Double
import numpy as np
import scipy.stats as st
import abc
import src.models as models


class OptimizerStrategy(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def update_model(self, X, y, model):
        """Implement Update Weigth Strategy"""


class SteepestDescentMethod(OptimizerStrategy, abc.ABC):
    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)

    def update_model(self, X, y, model):
        dedw = (X.T @ X @ model.w - X.T @ y)
        model.w = model.w - self.learning_rate * dedw


class NewtonsMethod(OptimizerStrategy, abc.ABC):
    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)

    def update_model(self, X, y, model):
        model.w = (1 - self.learning_rate) * model.w + self.learning_rate * (np.linalg.inv(np.dot(X.T, X))@X.T@y)
