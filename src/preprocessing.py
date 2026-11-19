"""Column filters for the preprocessing pipeline in main.ipynb."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DropAllZeroColumns(TransformerMixin, BaseEstimator):
    """Learn which measurement columns contain only zeros."""

    def fit(self, X, y=None):
        all_zero = X.min().eq(0) & X.max().eq(0)
        self.dropped_columns_ = X.columns[all_zero].tolist()
        self.retained_columns_ = X.columns[~all_zero].tolist()
        return self

    def transform(self, X):
        check_is_fitted(self, "retained_columns_")
        return X.loc[:, self.retained_columns_]


class SpearmanFilter(TransformerMixin, BaseEstimator):
    """Drop later measurements that strongly correlate with retained ones."""

    def __init__(self, *, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        correlations = X.corr(method="spearman").abs()
        retained = []
        removed = []

        for feature in X.columns:
            matches = correlations.loc[retained, feature]
            matches = matches[matches > self.threshold]
            if matches.empty:
                retained.append(feature)
            else:
                representative = matches.idxmax()
                removed.append(
                    {
                        "Removed feature": feature,
                        "Retained feature": representative,
                        "Absolute Spearman correlation": matches[representative],
                    }
                )

        self.retained_columns_ = retained
        self.removals_ = pd.DataFrame(
            removed,
            columns=[
                "Removed feature",
                "Retained feature",
                "Absolute Spearman correlation",
            ],
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "retained_columns_")
        return X.loc[:, self.retained_columns_]
