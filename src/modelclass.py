from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import sklearn.model_selection as sks

class ModelClass:
    def __init__(self, model, X, y, k=3, random_state=42,test_size=0.2):
        self.model = model
        self.X = X
        self.y = y
        self.k = k
        self.random_state = random_state
        self.skf = sks.StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = sks.train_test_split(
                    self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )

    def cross_validate(self):
        scores = sks.cross_val_score(self.model, self.X_train, self.y_train, cv=self.skf)
        return scores

    def hyperparameter_tuning(self, param_distributions, scoring='accuracy', cv=None):
        if cv is None:
            cv = self.skf
        self.param_distributions = param_distributions
        search = sks.RandomizedSearchCV(
            self.model,
            param_distributions, 
            scoring=scoring, 
            cv=cv,
            random_state=self.random_state, 
            n_jobs=1,
            n_iter=10,
        )
        search.fit(self.X_train, self.y_train)
        self.best_model_ = search.best_estimator_
        return search.best_estimator_, search.best_params_, search.best_score_

    def evaluate_model(self):
        y_pred = self.best_model_.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        return report

    def showvalues(self):
        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_train shape:", self.y_train.shape)
        print("y_test shape:", self.y_test.shape)

    def showvalidationcurves(self, scor="f1_macro"):
        if self.best_model_ is None or self.param_distributions is None:
            raise ValueError("Esegui prima hyperparameter_tuning()!")

        numeric_params = {}
        for param, values in self.param_distributions.items():
            valid_values = [
                v for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if len(valid_values) >= 2:
                numeric_params[param] = sorted(valid_values)

        if not numeric_params:
            raise ValueError("Nessun iperparametro numerico con almeno 2 valori da plottare.")

        modello_base = self.best_model_
        num_params = len(numeric_params)
        rows = (num_params + 1) // 2
        cols = 2 if num_params > 1 else 1

        figure, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes_flat = np.atleast_1d(axes).flatten()

        for i, (param, valid_values) in enumerate(numeric_params.items()):
            sks.ValidationCurveDisplay.from_estimator(
                modello_base,
                self.X_train,
                self.y_train,
                param_name=param,
                param_range=valid_values,
                scoring=scor,
                cv=self.skf,
                n_jobs=1,
                ax=axes_flat[i],
            )
            axes_flat[i].set_title(f"Validation Curve: {param}")
            axes_flat[i].grid(True, linestyle="--", alpha=0.5)

        for j in range(num_params, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()