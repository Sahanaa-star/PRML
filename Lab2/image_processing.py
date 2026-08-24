# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt


# def load_image(filename):
#     image = Image.open(filename)
#     gray = image.convert("L")
#     A = np.array(gray, dtype=float)
#     return image, gray, A

# def display_grayscale(gray):
#     plt.figure(figsize=(6, 6))
#     plt.imshow(gray, cmap="gray")
#     plt.title("Grayscale Image")
#     plt.axis("off")
#     plt.show()

# def perform_evd(A):
#     eigenvalues, Q = np.linalg.eig(A)
#     return eigenvalues, Q


# def select_eigen_components(eigenvalues, k):
#     order = np.argsort(np.abs(eigenvalues))[::-1]

#     selected = []
#     used = set()
#     tolerance = 1e-6

#     for index in order:

#         if index in used:
#             continue

#         value = eigenvalues[index]

#         if abs(value.imag) < tolerance:

#             if len(selected) < k:
#                 selected.append(index)
#                 used.add(index)

#         else:

#             conjugate_value = np.conjugate(value)
#             conjugate_index = None

#             for j in range(len(eigenvalues)):

#                 if j in used or j == index:
#                     continue

#                 if np.isclose(
#                     eigenvalues[j],
#                     conjugate_value,
#                     atol=tolerance
#                 ):
#                     conjugate_index = j
#                     break

#             if conjugate_index is not None:

#                 if len(selected) + 2 <= k:
#                     selected.append(index)
#                     selected.append(conjugate_index)

#                     used.add(index)
#                     used.add(conjugate_index)

#         if len(selected) >= k:
#             break

#     return selected


# def evd_rank_k(Q, eigenvalues, k):

#     n = len(eigenvalues)

#     selected_indices = select_eigen_components(eigenvalues, k)

#     Lambda_k = np.zeros((n, n), dtype=complex)

#     for index in selected_indices:
#         Lambda_k[index, index] = eigenvalues[index]

#     Q_inverse = np.linalg.inv(Q)

#     A_k = Q @ Lambda_k @ Q_inverse

#     A_k = np.real(A_k)

#     return A_k, len(selected_indices)


# def evd_error(A, A_k):
#     return np.linalg.norm(A - A_k, ord="fro")


# def perform_svd(A):

#     U, singular_values, Vt = np.linalg.svd(
#         A,
#         full_matrices=False
#     )

#     return U, singular_values, Vt


# def svd_rank_k(U, singular_values, Vt, k):

#     U_k = U[:, :k]

#     S_k = np.diag(singular_values[:k])

#     Vt_k = Vt[:k, :]

#     A_k = U_k @ S_k @ Vt_k

#     return A_k


# def svd_error(A, A_k):
#     return np.linalg.norm(A - A_k, ord="fro")

# image_processing.py

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
