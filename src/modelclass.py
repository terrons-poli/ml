from copy import deepcopy
from typing import Self, TypedDict, Unpack, Literal

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import sklearn.model_selection as sks
from sklearn.base import BaseEstimator
from numpy.typing import NDArray
import pandas as pd


class _ModelClassOverrides(TypedDict, total=False):
    model: BaseEstimator
    X: pd.DataFrame | NDArray
    y: pd.DataFrame | NDArray
    fold: sks.BaseCrossValidator
    random_state: int | np.random.RandomState | None
    test_size: float | int | None


class ModelClass:
    def __init__(
        self,
        model: BaseEstimator,
        X: pd.DataFrame | NDArray,
        y: pd.DataFrame | NDArray,
        fold: sks.BaseCrossValidator,
        random_state: int | np.random.RandomState | None = 42,
        test_size: float | int | None = 0.2,
    ):
        self.model = model
        self.X = X
        self.y = y
        self.random_state = random_state
        self.fold = fold
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = sks.train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

    @classmethod
    def with_stratified_fold(
        cls,
        model: BaseEstimator,
        X: pd.DataFrame | NDArray,
        y: pd.DataFrame | NDArray,
        k=3,
        random_state: int | np.random.RandomState | None = 42,
        test_size: float | int | None = 0.2,
    ):
        fold = sks.StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        return cls(model, X, y, fold, random_state, test_size)

    def copy_with(self, **changes: Unpack[_ModelClassOverrides]) -> Self:
        """Create a new experiment with the given constructor arguments replaced."""
        arguments: _ModelClassOverrides = {
            "model": self.model,
            "X": self.X,
            "y": self.y,
            "fold": self.fold,
            "random_state": deepcopy(self.random_state),
            "test_size": self.test_size,
        }
        arguments.update(changes)
        return type(self)(**arguments)

    def cross_validate(self):
        scores = sks.cross_val_score(
            self.model, self.X_train, self.y_train, cv=self.fold
        )
        return scores

    def hyperparameter_tuning(
        self,
        param_distributions: map,
        scoring="accuracy",
        n_jobs=1,
        n_iter=10,
        verbose=0,
        *,
        method: Literal["random", "grid"] = "random",
    ):
        """Tune on the training set and save the fitted search's CV results."""
        options = dict(scoring=scoring, cv=self.fold, n_jobs=n_jobs, verbose=verbose)

        if method == "random":
            search = sks.RandomizedSearchCV(
                self.model,
                param_distributions,
                random_state=self.random_state,
                n_iter=n_iter,
                **options,
            )
        elif method == "grid":
            search = sks.GridSearchCV(self.model, param_distributions, **options)
        else:
            raise ValueError(f"Unknown search method: {method!r}")

        search.fit(self.X_train, self.y_train)
        self.best_model = search.best_estimator_
        self.param_distributions = param_distributions
        self.cv_results_ = search.cv_results_

        return search.best_estimator_, search.best_params_, search.best_score_

    def evaluate_model(self):
        y_pred = self.best_model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        return report

    def show_values(self):
        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_train shape:", self.y_train.shape)
        print("y_test shape:", self.y_test.shape)

    def show_validation_curves(self, scoring="f1_macro"):
        if self.best_model is None or self.param_distributions is None:
            raise ValueError("Run hyperparameter_tuning() first.")

        numeric_params = {}
        for param, values in self.param_distributions.items():
            valid_values = [
                v
                for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if len(valid_values) >= 2:
                numeric_params[param] = sorted(valid_values)

        if not numeric_params:
            raise ValueError(
                "Plotting requires at least two values for a numeric hyperparameter."
            )

        num_params = len(numeric_params)
        rows = (num_params + 1) // 2
        cols = 2 if num_params > 1 else 1

        figure, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes_flat = np.atleast_1d(axes).flatten()

        for i, (param, valid_values) in enumerate(numeric_params.items()):
            sks.ValidationCurveDisplay.from_estimator(
                self.best_model,
                self.X_train,
                self.y_train,
                param_name=param,
                param_range=valid_values,
                scoring=scoring,
                cv=self.fold,
                n_jobs=1,
                ax=axes_flat[i],
            )
            axes_flat[i].set_title(f"Validation Curve: {param}")
            axes_flat[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(num_params, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()
