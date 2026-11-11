import numpy as np

class PCA:
    def fit(self, X):
        X = np.asarray(X, dtype=float)

        # Center using training means.
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_

        # Sample covariance matrix.
        covariance = (
            centered.T @ centered
        ) / (len(X) - 1)

        # Eigenvalues and eigenvectors of the covariance matrix.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Arrange from largest to smallest eigenvalue.
        order = np.argsort(eigenvalues)[::-1]

        self.eigenvalues_ = np.maximum(
            eigenvalues[order], 0.0
        )

        # Each column is a principal direction.
        self.components_ = eigenvectors[:, order]

        total_variance = self.eigenvalues_.sum()

        if total_variance <= 0:
            raise ValueError("PCA requires nonzero variance.")

        self.explained_variance_ratio_ = (
            self.eigenvalues_ / total_variance
        )

        self.cumulative_variance_ = np.cumsum(
            self.explained_variance_ratio_
        )

        return self

    def transform(self, X, n_components):
        if not 1 <= n_components <= self.components_.shape[1]:
            raise ValueError("Invalid number of PCA components.")

        centered = np.asarray(X, dtype=float) - self.mean_

        return centered @ self.components_[:, :n_components]