import numpy as np

def logsumexp_manual(values, axis=1):
    """Calculate log(sum(exp(values))) stably."""
    maximum = np.max(values, axis=axis, keepdims=True)

    result = maximum + np.log(
        np.sum(
            np.exp(values - maximum),
            axis=axis,
            keepdims=True
        )
    )

    return np.squeeze(result, axis=axis)


def log_gaussian_density(X, mean, covariance):
    """Calculate the Gaussian log density for each sample."""
    n_features = X.shape[1]

    # Difference between each sample and the Gaussian mean.
    centered = X - mean

    # Inverse covariance: Sigma^(-1).
    inverse_covariance = np.linalg.inv(covariance)

    # Calculate log|Sigma| directly for numerical stability.
    sign, log_determinant = np.linalg.slogdet(covariance)

    if sign <= 0:
        raise ValueError(
            "Covariance determinant must be positive."
        )

    # Calculate (x - mean)^T Sigma^(-1) (x - mean)
    # for each sample.
    squared_distance = np.sum(
        (centered @ inverse_covariance) * centered,
        axis=1
    )

    return -0.5 * (
        n_features * np.log(2 * np.pi)
        + log_determinant
        + squared_distance
    )

class GMM:
    def __init__(
        self,
        n_components=1,
        max_iter=150,
        tol=1e-5,
        regularization=1e-5
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization

    def _initialize(self, X):
        n_samples, n_features = X.shape

        if not 1 <= self.n_components <= n_samples:
            raise ValueError(
                "Invalid number of Gaussian components."
            )

        overall_mean = X.mean(axis=0)

        if self.n_components == 1:
            self.means_ = overall_mean[None, :].copy()

        else:
            # Choose the first center as the sample farthest
            # from the overall mean.
            distances_from_mean = np.sum(
                (X - overall_mean) ** 2,
                axis=1
            )

            first_index = int(np.argmax(distances_from_mean))
            chosen_indices = [first_index]

            nearest_squared_distance = np.sum(
                (X - X[first_index]) ** 2,
                axis=1
            )

            # Choose remaining centers farthest from the
            # already selected centers.
            while len(chosen_indices) < self.n_components:
                nearest_squared_distance[chosen_indices] = -np.inf

                next_index = int(
                    np.argmax(nearest_squared_distance)
                )

                if nearest_squared_distance[next_index] <= 0:
                    raise ValueError(
                        "Not enough distinct samples "
                        "for initialization."
                    )

                chosen_indices.append(next_index)

                new_distance = np.sum(
                    (X - X[next_index]) ** 2,
                    axis=1
                )

                nearest_squared_distance = np.minimum(
                    nearest_squared_distance,
                    new_distance
                )

            self.means_ = X[chosen_indices].copy()

        # Initialize each component with the overall covariance.
        centered = X - overall_mean

        initial_covariance = (
            centered.T @ centered / n_samples
            + self.regularization * np.eye(n_features)
        )

        self.covariances_ = np.repeat(
            initial_covariance[None, :, :],
            self.n_components,
            axis=0
        )

        # Initially, all components have equal weights.
        self.weights_ = np.full(
            self.n_components,
            1.0 / self.n_components
        )

    def _weighted_log_probabilities(self, X):
        """Return log(weight * Gaussian density) per component."""
        return np.column_stack([
            np.log(self.weights_[component])
            + log_gaussian_density(
                X,
                self.means_[component],
                self.covariances_[component]
            )
            for component in range(self.n_components)
        ])

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        self._initialize(X)

        self.log_likelihood_history_ = []
        self.converged_ = False

        previous_log_likelihood = None

        # Evaluate the initial model and every subsequent update.
        for iteration in range(self.max_iter + 1):
            weighted_log_probabilities = (
                self._weighted_log_probabilities(X)
            )

            sample_log_probabilities = logsumexp_manual(
                weighted_log_probabilities,
                axis=1
            )

            average_log_likelihood = np.mean(
                sample_log_probabilities
            )

            self.log_likelihood_history_.append(
                average_log_likelihood
            )

            # Stop when the average log likelihood stabilizes.
            if previous_log_likelihood is not None:
                change = abs(
                    average_log_likelihood
                    - previous_log_likelihood
                )

                if change < self.tol:
                    self.converged_ = True
                    break

            if iteration == self.max_iter:
                break

            # E-step: calculate membership probabilities.
            responsibilities = np.exp(
                weighted_log_probabilities
                - sample_log_probabilities[:, None]
            )

            effective_counts = responsibilities.sum(axis=0)

            if np.any(effective_counts < 1e-8):
                raise ValueError(
                    "A Gaussian component has negligible "
                    "membership. Try fewer components."
                )

            # M-step: update component weights.
            self.weights_ = effective_counts / n_samples

            # Update component means.
            self.means_ = (
                responsibilities.T @ X
            ) / effective_counts[:, None]

            # Update component covariance matrices.
            for component in range(self.n_components):
                centered = X - self.means_[component]

                weighted_centered = (
                    centered
                    * responsibilities[:, component, None]
                )

                covariance = (
                    weighted_centered.T @ centered
                ) / effective_counts[component]

                self.covariances_[component] = (
                    covariance
                    + self.regularization * np.eye(n_features)
                )

            previous_log_likelihood = average_log_likelihood

        self.n_iter_ = iteration

        return self

    def score_samples(self, X):
        """Return the GMM log density for each sample."""
        X = np.asarray(X, dtype=float)

        return logsumexp_manual(
            self._weighted_log_probabilities(X),
            axis=1
        )

class GMMClassifier:
    def __init__(
        self,
        n_components=1,
        max_iter=150,
        tol=1e-5,
        regularization=1e-5
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization

    def fit(self, X, y):
        self.classes_, counts = np.unique(
            y, return_counts=True
        )

        self.priors_ = counts / len(y)
        self.models_ = []

        for class_label in self.classes_:
            class_data = X[y == class_label]

            model = GMM(
                n_components=self.n_components,
                max_iter=self.max_iter,
                tol=self.tol,
                regularization=self.regularization
            )

            model.fit(class_data)
            self.models_.append(model)

        return self

    def predict(self, X):
        class_scores = np.column_stack([
            model.score_samples(X) + np.log(self.priors_[i])
            for i, model in enumerate(self.models_)
        ])

        predicted_indices = np.argmax(class_scores, axis=1)

        return self.classes_[predicted_indices]