import numpy as np
from scipy.optimize import minimize

def svm_objective_and_gradient(parameters, X, y_signed, C):
    # The last parameter is the intercept.
    weights = parameters[:-1]
    intercept = parameters[-1]

    scores = X @ weights + intercept

    # Positive loss when a sample is inside the margin
    # or on the wrong side of the decision boundary.
    hinge = np.maximum(
        0.0,
        1.0 - y_signed * scores
    )

    # Squared-hinge SVM objective.
    objective = (
        0.5 * np.dot(weights, weights)
        + C * np.mean(hinge ** 2)
    )

    # Derivative of the loss with respect to each score.
    score_gradient = (
        -2.0 * C * y_signed * hinge / len(y_signed)
    )

    weight_gradient = weights + X.T @ score_gradient
    intercept_gradient = np.sum(score_gradient)

    gradient = np.concatenate([
        weight_gradient,
        [intercept_gradient]
    ])

    return objective, gradient

class LinearSVM:
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError(
                "This SVM implementation requires two classes."
            )

        # First class -> -1; second class -> +1.
        y_signed = np.where(
            y == self.classes_[0], -1.0, 1.0
        )

        # Deterministic initialization: all zeros.
        initial_parameters = np.zeros(X.shape[1] + 1)

        self.optimization_result_ = minimize(
            fun=svm_objective_and_gradient,
            x0=initial_parameters,
            args=(X, y_signed, self.C),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": 1e-12,
                "gtol": 1e-6
            }
        )

        self.weights_ = self.optimization_result_.x[:-1]
        self.intercept_ = self.optimization_result_.x[-1]

        self.converged_ = self.optimization_result_.success
        self.n_iter_ = self.optimization_result_.nit

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)

        return X @ self.weights_ + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)

        return np.where(
            scores >= 0,
            self.classes_[1],
            self.classes_[0]
        )

class OneVsRestSVM:
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.models_ = []

        for digit in self.classes_:
            # Current digit = 1; all other digits = 0.
            binary_labels = (y == digit).astype(int)

            model = LinearSVM(
                C=self.C,
                max_iter=self.max_iter
            )

            model.fit(X, binary_labels)
            self.models_.append(model)

        self.converged_ = all(
            model.converged_
            for model in self.models_
        )

        return self

    def decision_function(self, X):
        return np.column_stack([
            model.decision_function(X)
            for model in self.models_
        ])

    def predict(self, X):
        scores = self.decision_function(X)

        return self.classes_[
            np.argmax(scores, axis=1)
        ]