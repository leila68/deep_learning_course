import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_CIFAR10(data_path):
    """
    Load CIFAR-10 dataset.

    Parameters:
    - data_path: Path to the CIFAR-10 dataset.

    Returns:
    - Xtr: Training data
    - Ytr: Training labels
    - Xte: Test data
    - Yte: Test labels
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Flatten labels to 1D arrays
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    return x_train, y_train, x_test, y_test

# Load CIFAR-10 data
Xtr, Ytr, Xte, Yte = load_CIFAR10('C:/Users/Leila/Downloads/cifar-10-python/cifar-10-batches-py/')

# Flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], -1)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], -1)  # Xte_rows becomes 10000 x 3072

# Print the shapes
print('Xtr_rows shape:', Xtr_rows.shape)
print('Ytr shape:', Ytr.shape)
print('Xte_rows shape:', Xte_rows.shape)
print('Yte shape:', Yte.shape)

# Create a k-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training images and labels
knn.fit(Xtr_rows, Ytr)

# Predict labels on the test images
Yte_predict = knn.predict(Xte_rows[:10, :])  # Predict for the first 10 test examples

# Display the images and predictions
num_rows_to_show = 3
image_height, image_width, num_channels = 32, 32, 3

images_to_show = Xte_rows[:10, :].reshape((-1, image_height, image_width, num_channels))

plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images_to_show[i].astype('uint8'))
    plt.title(f"True: {Yte[i]}, Predicted: {Yte_predict[i]}")
    plt.axis("off")

plt.show()

# Print the classification accuracy
accuracy = accuracy_score(Yte[:10], Yte_predict)  # Accuracy for the first 10 test examples
print('Accuracy:', accuracy)
