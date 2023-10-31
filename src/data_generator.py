import numpy as np
import scipy.stats as st


class DataGenerator:
    def __init__(self, sample_size, weights, x_min, x_max) -> None:
        self.sample_size = sample_size
        self.weights = weights
        self.x_min = x_min
        self.x_max = x_max

    def get_data(self):
        x_0 = np.ones((self.sample_size, 1))
        x_1 = np.array(
            np.linspace(start=self.x_min, stop=self.x_max, num=self.sample_size).reshape(self.sample_size, 1))
        X = np.column_stack([x_0, x_1])
        Y = np.array(np.dot(X, self.weights))
        E = np.array(st.norm.rvs(loc=0, scale=1, size=self.sample_size).reshape(1, self.sample_size)).T
        Y = np.add(Y, E)
        return x_1, Y
