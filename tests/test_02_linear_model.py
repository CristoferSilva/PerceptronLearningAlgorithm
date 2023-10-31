import unittest
import numpy as np
from tests.test_base_mixin import TestBaseMixin
from src.models import LinearModel
from src.preprocessing import Preprocessing

class TestLinearModel(TestBaseMixin):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        
    def setUp(self) -> None:
        super().setUp()
        self.model = LinearModel()

    def test_linear_model_zero_weights(self):
        self.model.w = np.array([[0,0]]).T
        self.Xd = Preprocessing.build_design_matrix(self.X)
        yhat = self.model.predict(self.Xd)
        errors = yhat - self.y
        actual = 1.0/len(self.Xd) * np.square(np.linalg.norm(errors))     
        expected = 1.0/len(self.Xd) * np.square(np.linalg.norm(self.y))     
        self.assertAlmostEqual(expected, actual, delta=0.1)

    def test_linear_model_perfect_weights(self):
        self.model.w = self.weights
        self.Xd = Preprocessing.build_design_matrix(self.X)
        yhat = self.model.predict(self.Xd)
        errors = yhat - self.y
        actual = 1.0/len(self.Xd) * np.square(np.linalg.norm(errors))     
        expected = 1
        self.assertAlmostEqual(expected, actual, delta=0.1)

    

if __name__ == '__main__':
    unittest.main()