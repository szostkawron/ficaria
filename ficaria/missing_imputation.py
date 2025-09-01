from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Example class:

class SimpleMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # No parameters needed for now, but you can add them here if needed
        pass

    def fit(self, X, y=None):
        # Assume X is a DataFrame or convertible to one
        X = self._check_input(X)
        # Calculate the mean of each column ignoring NaNs
        self.means_ = X.mean()
        return self  # required for sklearn compatibility

    def transform(self, X):
        X = self._check_input(X)
        # Fill NaNs with the column means computed in fit
        return X.fillna(self.means_)

    def _check_input(self, X):
        # Helper to ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X
