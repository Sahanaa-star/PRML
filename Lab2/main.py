# import numpy as np
# import matplotlib.pyplot as plt

# from image_processing import (
#     load_image,
#     display_grayscale,
#     perform_evd,
#     evd_rank_k,
#     evd_error,
#     perform_svd,
#     svd_rank_k,
#     svd_error
# )


# filename = "cat_08.jpg"

# image, gray, A = load_image(filename)

# print("Original image mode:", image.mode)
# print("Grayscale image mode:", gray.mode)
# print("Matrix shape:", A.shape)

# display_grayscale(gray)

# print("\nPerforming EVD...")

# eigenvalues, Q = perform_evd(A)

# print("EVD completed.")
# print("Number of eigenvalues:", len(eigenvalues))


# order = np.argsort(np.abs(eigenvalues))[::-1]
# eigenvalues = eigenvalues[order]
# Q = Q[:, order]


# print("\nFirst 10 eigenvalues:")

# for i in range(10):
#     print(i + 1, eigenvalues[i])


# complex_values = [
#     value for value in eigenvalues
#     if abs(value.imag) > 1e-6
# ]

# print("\nNumber of complex eigenvalues:", len(complex_values))

# print("\nComplex eigenvalues:")

# shown = 0

# for value in complex_values:

#     if shown >= 10:
#         break

#     conjugate_value = np.conjugate(value)

#     print(value, "<->", conjugate_value)

#     shown += 1


# Lambda = np.diag(eigenvalues)

# A_full = Q @ Lambda @ np.linalg.inv(Q)

# A_full = np.real(A_full)

# full_error = evd_error(A, A_full)

# print("\nFull EVD reconstruction error:", full_error)


# k_values = [10, 50, 100]

# evd_errors = []
# svd_errors = []


# for k in k_values:

#     A_evd, actual_k = evd_rank_k(
#         Q,
#         eigenvalues,
#         k
#     )

#     error = evd_error(A, A_evd)

#     evd_errors.append(error)

#     print("\nEVD")
#     print("Requested k:", k)
#     print("Actual components retained:", actual_k)
#     print("Frobenius error:", error)

#     error_image = np.abs(A - A_evd)

#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 3, 1)
#     plt.imshow(A, cmap="gray")
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.imshow(
#         np.clip(A_evd, 0, 255),
#         cmap="gray"
#     )
#     plt.title(f"EVD Reconstruction k={actual_k}")
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.imshow(error_image, cmap="gray")
#     plt.title(f"EVD Error Image k={actual_k}")
#     plt.axis("off")

#     plt.show()


# print("\nPerforming SVD...")

# U, singular_values, Vt = perform_svd(A)

# print("SVD completed.")
# print("Number of singular values:", len(singular_values))


# for k in k_values:

#     A_svd = svd_rank_k(
#         U,
#         singular_values,
#         Vt,
#         k
#     )

#     error = svd_error(A, A_svd)

#     svd_errors.append(error)

#     print("\nSVD")
#     print("k:", k)
#     print("Frobenius error:", error)

#     error_image = np.abs(A - A_svd)

#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 3, 1)
#     plt.imshow(A, cmap="gray")
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.imshow(
#         np.clip(A_svd, 0, 255),
#         cmap="gray"
#     )
#     plt.title(f"SVD Reconstruction k={k}")
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.imshow(error_image, cmap="gray")
#     plt.title(f"SVD Error Image k={k}")
#     plt.axis("off")

#     plt.show()


# plt.figure(figsize=(8, 5))

# plt.plot(
#     k_values,
#     evd_errors,
#     marker="o",
#     label="EVD"
# )

# plt.plot(
#     k_values,
#     svd_errors,
#     marker="o",
#     label="SVD"
# )

# plt.xlabel("Number of retained components (k)")
# plt.ylabel("Frobenius Reconstruction Error")
# plt.title("EVD vs SVD Reconstruction Error")

# plt.legend()

# plt.grid()

# plt.show()

# main.py

import numpy as np
import matplotlib.pyplot as plt

from image_processing import (
    load_image,
    display_grayscale,
    perform_evd,
    evd_rank_k,
    evd_error,
    perform_svd,
    svd_rank_k,
    svd_error
)


IMAGE_NAME = "cat_08.jpg"
COMPONENTS = [10, 50, 100]


def show_image_comparison(original, approximation, method, k):
    """
    Display the original image, its reconstruction,
    and the difference between them.
    """

    difference = np.abs(original - approximation)

    images = [
        original,
        np.clip(approximation, 0, 255),
        difference
    ]

    titles = [
        "Original Image",
        f"{method} Reconstruction (k={k})",
        f"{method} Reconstruction Error"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, image_data, title in zip(axes, images, titles):
        axis.imshow(image_data, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def print_eigenvalue_details(eigenvalues):
    """Display information about the computed eigenvalues."""

    ranked_values = eigenvalues[
        np.argsort(np.abs(eigenvalues))[::-1]
    ]

    print("\nFirst 10 eigenvalues:")

    for position, value in enumerate(
        ranked_values[:10],
        start=1
    ):
        print(f"{position}: {value}")

    complex_mask = np.abs(
        ranked_values.imag
    ) > 1e-6

    complex_values = ranked_values[complex_mask]

    print(
        "\nNumber of complex eigenvalues:",
        len(complex_values)
    )

    print("\nSome complex eigenvalue conjugate pairs:")

    for value in complex_values[:10]:
        print(
            f"{value}  <->  {np.conjugate(value)}"
        )


def run_evd(matrix):
    """Perform EVD and evaluate different rank-k approximations."""

    print("\n========== EVD ==========")

    eigenvalues, eigenvectors = perform_evd(matrix)

    print("EVD completed")
    print(
        "Number of eigenvalues:",
        eigenvalues.size
    )

    print_eigenvalue_details(eigenvalues)

    # Full reconstruction using all eigen-components
    diagonal = np.diag(eigenvalues)

    complete_reconstruction = (
        eigenvectors
        @ diagonal
        @ np.linalg.inv(eigenvectors)
    )

    complete_reconstruction = np.real_if_close(
        complete_reconstruction
    )

    full_error = evd_error(
        matrix,
        complete_reconstruction.real
    )

    print(
        "\nFull EVD reconstruction error:",
        full_error
    )

    errors = []

    for k in COMPONENTS:

        reconstruction, retained = evd_rank_k(
            eigenvectors,
            eigenvalues,
            k
        )

        current_error = evd_error(
            matrix,
            reconstruction
        )

        errors.append(current_error)

        print(f"\nRank-{retained} EVD approximation")
        print("Requested components:", k)
        print("Components used:", retained)
        print("Frobenius error:", current_error)

        show_image_comparison(
            matrix,
            reconstruction,
            "EVD",
            retained
        )

    return errors


def run_svd(matrix):
    """Perform SVD and evaluate different rank-k approximations."""

    print("\n========== SVD ==========")

    left_vectors, singular_values, right_vectors = perform_svd(
        matrix
    )

    print("SVD completed")
    print(
        "Number of singular values:",
        singular_values.size
    )

    errors = []

    for k in COMPONENTS:

        reconstruction = svd_rank_k(
            left_vectors,
            singular_values,
            right_vectors,
            k
        )

        current_error = svd_error(
            matrix,
            reconstruction
        )

        errors.append(current_error)

        print(f"\nRank-{k} SVD approximation")
        print("Frobenius error:", current_error)

        show_image_comparison(
            matrix,
            reconstruction,
            "SVD",
            k
        )

    return errors


def plot_error_comparison(evd_errors, svd_errors):
    """Compare reconstruction errors from EVD and SVD."""

    fig, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        COMPONENTS,
        evd_errors,
        marker="o",
        label="EVD"
    )

    axis.plot(
        COMPONENTS,
        svd_errors,
        marker="o",
        label="SVD"
    )

    axis.set_xlabel("Retained Components (k)")
    axis.set_ylabel("Frobenius Norm Error")
    axis.set_title("Comparison of EVD and SVD Reconstruction Errors")

    axis.legend()
    axis.grid(True)

    plt.tight_layout()
    plt.show()


def main():

    image, gray_image, matrix = load_image(
        IMAGE_NAME
    )

    print("Original image mode:", image.mode)
    print("Grayscale image mode:", gray_image.mode)
    print("Image matrix size:", matrix.shape)

    display_grayscale(gray_image)

    if matrix.shape[0] != matrix.shape[1]:
        print("\nEVD requires a square image matrix.")
        return

    evd_errors = run_evd(matrix)

    svd_errors = run_svd(matrix)

    plot_error_comparison(
        evd_errors,
        svd_errors
    )


if __name__ == "__main__":
    main()

