import unittest
import numpy as np
from src.data_generator import DataGenerator
from tests.test_base_mixin import TestBaseMixin


class TestDataGenerator(TestBaseMixin):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        
    def setUp(self) -> None:
        return super().setUp()
   
    def test_x_shape(self):
        expected = (super().sample_size, 1)
        actual = self.X.shape
        self.assertEqual(actual, expected)

    def test_y_shape(self):
        expected = (super().sample_size, 1)
        actual = self.y.shape
        self.assertEqual(actual, expected)

    def test_x_min(self):
        expected = self.x_min
        actual = np.min(self.X)
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_y_min(self):
        expected = -0.23
        actual = np.min(self.y)
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_x_max(self):
        expected = self.x_max
        actual = np.max(self.X)
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_y_max(self):
        expected = 19.86
        actual = np.max(self.y)
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_x_mean_at_1st_quarter(self):
        expected = -1.50
        actual = np.mean(self.X[0:250])
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_y_mean_at_1st_quarter(self):
        expected = 4.02
        actual = np.mean(self.y[0:250])
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_x_mean_at_2nd_quarter(self):
        expected = -0.50
        actual = np.mean(self.X[250:500])
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_y_mean_at_2nd_quarter(self):
        expected = 7.91
        actual = np.mean(self.y[250:500])
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_x_mean_at_3rd_quarter(self):
        expected = 0.50
        actual = np.mean(self.X[500:750])
        self.assertAlmostEqual(actual, expected, delta=0.1)

    def test_x_mean_at_4th_quarter(self):
        expected = 16.03
        actual = np.mean(self.y[750:1000])
        self.assertAlmostEqual(actual, expected, delta=0.1)

    

if __name__ == '__main__':
    unittest.main()