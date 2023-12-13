"""
Emre KARATAŞ
22001641
CS 464-Section 01
"""
import numpy as np
import gzip
import matplotlib.pyplot as plt


def read_pixels(data_path):
    """
    Read pixel data from a given file path.

    :param data_path: The path to the data file.
    :type data_path: str
    :return: The flattened pixel data.
    :rtype: np.ndarray
    """
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28 * 28)
    return flattened_pixels


def read_labels(data_path):
    """
    Read label data from a gzip file.

    :param data_path: The path to the gzip file containing the label data.
    :return: An array of label data.
    """
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    return label_data


images = read_pixels("data/train-images-idx3-ubyte.gz")
labels = read_labels("data/train-labels-idx1-ubyte.gz")


def pca(X, number_of_components):
    """
    Perform Principal Component Analysis (PCA) on the given dataset.

    :param X: The dataset in the form of a 2D numpy array.
    :param number_of_components: The number of principal components to retain.
    :return: A tuple containing the transformed dataset, the selected principal components, and the corresponding eigenvalues.

    """
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_components = eigenvectors[:, sorted_indices[:number_of_components]]
    X_pca = np.dot(X_centered, selected_components)
    return X_pca, selected_components, eigenvalues[sorted_indices]


# Apply PCA for 10 components
num_components = 10
pca_images, components, sorted_eigenvalues = pca(images, num_components)

# Calculate and Report Proportion of Variance Explained (PVE)
variance_explained = [(i / sum(sorted_eigenvalues)) * 100 for i in sorted_eigenvalues[:num_components]]
print("Proportion of Variance Explained by the first 10 principal components:")
for i, variance in enumerate(variance_explained):
    print(f"Component {i + 1}: {variance:.2f}%")


def threshold(th, pve):
    """
    Calculate the index and cumulative value at which the cumulative sum of the given data
    exceeds or equals the specified threshold.

    :param th: The threshold value.
    :type th: float

    :param pve: The data containing the percentage of explained variance.
    :type pve: numpy.ndarray

    :return: The index and cumulative value.
    :rtype: tuple

    :raises IndexError: If the threshold value is greater than the cumulative sum of pve.

    """
    cumulative_pve = np.cumsum(pve)
    th_index = np.argwhere(cumulative_pve >= th)
    return th_index[0][0] + 1, cumulative_pve[th_index[0][0]]


# Calculate the Proportion of Variance Explained
variance_explained = [(i / sum(sorted_eigenvalues)) * 100 for i in sorted_eigenvalues]

# Use the threshold function to find the number of components for 70% variance explained
num_components_70_percent, pve_at_70_percent = threshold(70, variance_explained)

print(f"Number of components to explain at least 70% of the data: {num_components_70_percent}")
print(f"Cumulative variance explained at this point: {pve_at_70_percent}%")


def visualize_components(components):
    """
    :param components: numpy array with shape (n_samples, n_components)
    :return: None

    Visualizes the given components in a grid format. Each component is displayed as an image.

    """
    fig, axes = plt.subplots(2, 5, figsize=(24, 6))  # 2 rows, 5 columns

    for i, ax in enumerate(axes.flatten()):
        if i < components.shape[1]:
            pc = components[:, i].reshape(28, 28)
            # Apply min-max scaling
            pc_scaled = (pc - np.min(pc)) / (np.max(pc) - np.min(pc))
            ax.imshow(pc_scaled, cmap='Greys_r')
            ax.set_title(f"{i + 1}")
            ax.axis('off')  # Remove axes for clarity
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks

    plt.show()


visualize_components(components)

# Selecting the first 100 images and their labels
first_100_images = images[:100]
first_100_labels = labels[:100]

# Projecting onto the first 2 principal components
pca_projection = np.dot(first_100_images - np.mean(first_100_images, axis=0), components[:, :2])

# Plotting the projected data points colored by labels
plt.figure(figsize=(8, 6))
for i in range(10):  # Looping through each digit label (0-9)
    idx = first_100_labels == i
    plt.scatter(pca_projection[idx, 0], pca_projection[idx, 1], label=i)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Projection of First 100 MNIST Images onto First 2 Principal Components')
plt.show()


def reconstruct_image(image, components, mean, k):
    """

    Reconstructs an image from its principal components.

    :param image: The input image to be reconstructed.
    :param components: The matrix containing the principal components.
    :param mean: The mean image used for centering.
    :param k: The number of principal components to be used.

    :return: The reconstructed image.

    """
    image_centered = image - mean
    projected = np.dot(image_centered, components[:, :k])
    reconstructed = np.dot(projected, components[:, :k].T) + mean
    return reconstructed


mean_image = np.mean(images, axis=0)
first_image = images[0]

k_values = [1, 50, 100, 250, 500, 784]
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for i, k in enumerate(k_values):
    # Reconstruct the image using k principal components
    reconstructed_image = reconstruct_image(first_image, components, mean_image, k).reshape(28, 28)

    # Plot the reconstructed image
    ax = axes[i // 3, i % 3]
    ax.imshow(reconstructed_image, cmap='gray')
    ax.set_title(f'{k} Components')
    ax.axis('off')  # Hide the axis labels and ticks
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

plt.tight_layout()
plt.show()
