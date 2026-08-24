import matplotlib.pyplot as plt
import numpy as np

from image_processing import load_image, perform_svd, svd_rank_k, svd_error


IMAGE_FILE = "cat_08 (1).png"
RETAINED_RANKS = (10, 50, 100)


def show_reconstruction(original, reconstructed, rank):
    difference = np.abs(original - reconstructed)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(reconstructed, 0, 255), cmap="gray")
    axes[1].set_title(f"SVD, k = {rank}")
    axes[1].axis("off")

    axes[2].imshow(difference, cmap="gray")
    axes[2].set_title(f"Absolute Error, k = {rank}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():

    image, grayscale, matrix = load_image(IMAGE_FILE)

    print("Original image mode:", image.mode)
    print("Grayscale image mode:", grayscale.mode)
    print("Matrix dimensions:", matrix.shape)

    print("\nComputing SVD...")

    left_vectors, values, right_vectors = perform_svd(matrix)

    print("SVD completed successfully.")
    print("Total singular values:", len(values))

    reconstruction_errors = []

    for rank in RETAINED_RANKS:

        approximation = svd_rank_k(
            left_vectors,
            values,
            right_vectors,
            rank
        )

        error = svd_error(matrix, approximation)
        reconstruction_errors.append(error)

        print(f"\nRank k = {rank}")
        print(f"Frobenius reconstruction error = {error}")

        show_reconstruction(
            matrix,
            approximation,
            rank
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        RETAINED_RANKS,
        reconstruction_errors,
        "o-"
    )

    plt.xlabel("Retained rank (k)")
    plt.ylabel("Frobenius norm error")
    plt.title("SVD Reconstruction Error for Different Ranks")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
