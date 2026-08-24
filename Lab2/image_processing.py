import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image(filename):
    """Open an image and convert it into a grayscale matrix."""
    image = Image.open(filename)
    gray_image = image.convert("L")
    matrix = np.asarray(gray_image, dtype=np.float64)

    return image, gray_image, matrix


def display_grayscale(gray):
    """Show the grayscale version of the image."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(gray, cmap="gray")
    ax.set_title("Grayscale Image")
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def perform_evd(A):
    """Compute the eigenvalues and eigenvectors of a square matrix."""
    values, vectors = np.linalg.eig(A)

    return values, vectors


def select_eigen_components(eigenvalues, k):
    """
    Select dominant eigenvalue indices.

    Complex eigenvalues are selected together with their
    conjugate partners when possible.
    """
    ranked = np.argsort(-np.abs(eigenvalues))

    chosen = []
    processed = np.zeros(len(eigenvalues), dtype=bool)
    tolerance = 1e-6

    for index in ranked:

        if processed[index]:
            continue

        current = eigenvalues[index]

        # Real eigenvalue
        if np.isclose(current.imag, 0, atol=tolerance):

            if len(chosen) < k:
                chosen.append(index)
                processed[index] = True

        # Complex eigenvalue
        else:

            conjugate = np.conjugate(current)

            possible_matches = np.where(
                np.isclose(
                    eigenvalues,
                    conjugate,
                    atol=tolerance
                )
            )[0]

            partner = None

            for candidate in possible_matches:

                if candidate != index and not processed[candidate]:
                    partner = candidate
                    break

            if partner is not None and len(chosen) <= k - 2:

                chosen.extend([index, partner])

                processed[index] = True
                processed[partner] = True

        if len(chosen) >= k:
            break

    return chosen


def evd_rank_k(Q, eigenvalues, k):
    """
    Reconstruct the matrix using the selected eigen-components.
    """

    selected = select_eigen_components(
        eigenvalues,
        k
    )

    inverse_q = np.linalg.inv(Q)

    reconstruction = np.zeros(
        (Q.shape[0], Q.shape[0]),
        dtype=np.complex128
    )

    # Build the reconstruction by adding one eigen-component at a time
    for index in selected:

        eigen_component = (
            eigenvalues[index]
            * np.outer(
                Q[:, index],
                inverse_q[index, :]
            )
        )

        reconstruction += eigen_component

    reconstruction = np.real_if_close(
        reconstruction
    )

    return reconstruction.real, len(selected)


def evd_error(A, A_k):
    """Return the Frobenius norm of the EVD reconstruction error."""
    difference = A - A_k

    return np.linalg.norm(
        difference,
        "fro"
    )


def perform_svd(A):
    """Calculate the reduced Singular Value Decomposition."""

    left_matrix, singular_values, right_matrix = np.linalg.svd(
        A,
        full_matrices=False
    )

    return left_matrix, singular_values, right_matrix


def svd_rank_k(U, singular_values, Vt, k):
    """
    Construct a rank-k approximation from the first k
    singular components.
    """

    valid_k = min(k, len(singular_values))

    left_part = U[:, :valid_k]
    values_part = singular_values[:valid_k]
    right_part = Vt[:valid_k]

    # Each column of U is scaled by its corresponding singular value
    reconstruction = (
        left_part * values_part
    ) @ right_part

    return reconstruction


def svd_error(A, A_k):
    """Return the Frobenius norm of the SVD reconstruction error."""

    residual = np.subtract(A, A_k)

    return np.linalg.norm(
        residual,
        ord="fro"
    )
