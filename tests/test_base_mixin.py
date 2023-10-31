import unittest
from abc import ABC, abstractmethod
import numpy as np
from src.data_generator import DataGenerator

class TestBaseMixin(unittest.TestCase, ABC):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        np.random.seed(0)
        super().setUp()
        w0 = 10
        w1 = 4
        self._sample_size = 1000
        self.weights = np.array([[w0, w1]]).T
        self.x_min = -2
        self.x_max = 2
        self.generate_data()
        
        
    def generate_data(self):
        self.data_generator = DataGenerator(self.sample_size, self.weights, self.x_min, self.x_max )
        X, y = self.data_generator.get_data()
        self._X = X
        self._y = y

        
    @property
    def sample_size(self):
        return self._sample_size
    
    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
