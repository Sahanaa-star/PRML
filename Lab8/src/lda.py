import numpy as np

class LDA:
    def __init__(self, regularization=1e-6):
        self.regularization = regularization

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        pooled_scatter = np.zeros((n_features, n_features))

        for i, class_label in enumerate(self.classes_):
            class_data = X[y == class_label]

            self.means_[i] = class_data.mean(axis=0)
            self.priors_[i] = len(class_data) / n_samples

            centered = class_data - self.means_[i]
            pooled_scatter += centered.T @ centered

        covariance = pooled_scatter / (n_samples - n_classes)

        # Small diagonal adjustment for numerical stability.
        covariance += (
            self.regularization * np.eye(n_features)
        )

        # Solve covariance @ weights = class means.
        self.weights_ = np.linalg.solve(
            covariance, self.means_.T
        )

        self.intercepts_ = (
            -0.5 * np.sum(self.means_.T * self.weights_, axis=0)
            + np.log(self.priors_)
        )

        return self

    def predict(self, X):
        scores = X @ self.weights_ + self.intercepts_
        best_class_indices = np.argmax(scores, axis=1)

        return self.classes_[best_class_indices]