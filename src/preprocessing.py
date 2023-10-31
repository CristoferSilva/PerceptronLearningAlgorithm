import numpy as np
import scipy.stats as st


class Preprocessing:

    @staticmethod
    def build_design_matrix(X):
        x_0 = np.ones((X.shape[0], X.shape[1]))
        X_final = np.column_stack([x_0, X])
        return X_final

    @staticmethod
    def initilize_weights(X):
        m, n = X.shape
        errors = st.norm.rvs(size=(n + 1, 1))
        return np.zeros((n + 1, 1)) + errors
