import unittest
import numpy as np
from tests.test_base_mixin import TestBaseMixin
from src.models import LinearModel
from src.optimizers import SteepestDescentMethod, NewtonsMethod
from src.preprocessing import Preprocessing

class TestOptimizer(TestBaseMixin):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        
    def setUp(self) -> None:
        super().setUp()
        self.model = LinearModel()

    def test_steepest_descent_zero_weight(self):
        learning_rate = .001
        optimizer = SteepestDescentMethod(learning_rate)
        Xd = Preprocessing.build_design_matrix(self.X)
        model = LinearModel()
        model.w = np.array([[0,0]]).T
        expected = -learning_rate*(np.linalg.multi_dot([Xd.T, Xd, model.w]) - np.dot(Xd.T,self.y))
        optimizer.update_model(Xd, self.y, model)
        actual = model.w
        np.testing.assert_allclose(expected, actual, 0.1)

    def test_steepest_descent_one_weight(self):
        learning_rate = .001
        optimizer = SteepestDescentMethod(learning_rate)
        Xd = Preprocessing.build_design_matrix(self.X)
        model = LinearModel()
        model.w = np.array([[1,1]]).T
        expected = model.w - learning_rate*(np.linalg.multi_dot([Xd.T, Xd, model.w]) - np.dot(Xd.T,self.y))
        optimizer.update_model(Xd, self.y, model)
        actual = model.w
        np.testing.assert_allclose(expected, actual, 0.1)


    def test_newton_method_zero_weight(self):
        learning_rate = .001
        optimizer = NewtonsMethod(learning_rate)
        Xd = Preprocessing.build_design_matrix(self.X)
        model = LinearModel()
        model.w = np.array([[0,0]]).T
        expected = learning_rate *  np.linalg.multi_dot([np.linalg.inv(np.dot(Xd.T, Xd)), Xd.T, self.y])
        optimizer.update_model(Xd, self.y, model)
        actual = model.w
        np.testing.assert_allclose(expected, actual, 0.1)

    def test_newton_method_one_weight(self):
        learning_rate = .001
        optimizer = NewtonsMethod(learning_rate)
        Xd = Preprocessing.build_design_matrix(self.X)
        model = LinearModel()
        model.w = np.array([[1,1]]).T
        expected = (1 - learning_rate)* model.w + learning_rate *  np.linalg.multi_dot([np.linalg.inv(np.dot(Xd.T, Xd)), Xd.T, self.y])
        optimizer.update_model(Xd, self.y, model)
        actual = model.w
        np.testing.assert_allclose(expected, actual, 0.1)
        

if __name__ == '__main__':
    unittest.main()