import numpy as np

class KNN:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)

        self.classes_, self.encoded_y_ = np.unique(
            y, return_inverse=True
        )

        self.train_squared_norms_ = np.sum(
            self.X_train_ ** 2, axis=1
        )

        return self

    def predict_for_k_values(self, X, k_values):
        X = np.asarray(X, dtype=float)

        k_values = sorted(set(k_values))

        if not k_values or any(
            k < 1 or k > len(self.X_train_) for k in k_values
        ):
            raise ValueError(
                "Each k must be between 1 and the training sample count."
            )

        max_k = max(k_values)

        predictions = {
            k: np.empty(len(X), dtype=self.classes_.dtype)
            for k in k_values
        }

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch = X[start:end]

            # Squared Euclidean distances.
            distances = (
                np.sum(batch ** 2, axis=1)[:, None]
                + self.train_squared_norms_[None, :]
                - 2 * batch @ self.X_train_.T
            )

            distances = np.maximum(distances, 0.0)

            # Stable sorting makes equal-distance ordering deterministic.
            nearest_indices = np.argsort(
                distances, axis=1, kind="stable"
            )[:, :max_k]

            nearest_labels = self.encoded_y_[nearest_indices]

            for k in k_values:
                votes = np.column_stack([
                    np.sum(
                        nearest_labels[:, :k] == class_index,
                        axis=1
                    )
                    for class_index in range(len(self.classes_))
                ])

                predicted_indices = np.argmax(votes, axis=1)

                predictions[k][start:end] = (
                    self.classes_[predicted_indices]
                )

        return predictions