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

