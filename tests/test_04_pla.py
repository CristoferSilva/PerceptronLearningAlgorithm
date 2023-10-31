import unittest
import numpy as np
from tests.test_base_mixin import TestBaseMixin
from src.algorithms import PLA
from src.models import LinearModel
from src.optimizers import SteepestDescentMethod, NewtonsMethod
from src.analyzers import PlotterAlgorithmObserver
from src.stop_criteria import CompositeStopCriteria, MaxIterationStopCriteria, MinErrorStopCriteria


class TestPLA(TestBaseMixin):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(seed=0)
        self.model = LinearModel()
        min_error = 1
        max_iteration = 100
        self.stop_criteria = CompositeStopCriteria()
        self.stop_criteria.add(MinErrorStopCriteria(min_error=min_error))
        self.stop_criteria.add(MaxIterationStopCriteria(max_iteration=max_iteration))
   
    def test_steepest_descent_method(self):
        learning_rate = 0.001
        self.optimizer = SteepestDescentMethod(learning_rate)
        self.alg = PLA(self.optimizer, self.model)
        self.alg.fit(self.X, self.y, self.stop_criteria)
        expected = 0.994
        actual = self.alg.rmse
        self.assertAlmostEqual(expected, actual, delta=0.01)

    def test_steepest_newton_method(self):
        learning_rate = 0.1
        self.optimizer = NewtonsMethod(learning_rate)
        self.alg = PLA(self.optimizer, self.model)
        self.alg.fit(self.X, self.y, self.stop_criteria)
        expected = 0.994
        actual = self.alg.rmse
        self.assertAlmostEqual(expected, actual, delta=0.01)
    

if __name__ == '__main__':
    unittest.main()