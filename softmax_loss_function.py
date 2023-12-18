import numpy as np


def softmax_cross_entropy_loss(X, y):
    """
    Softmax Cross-Entropy loss function.

    Parameters:
    - X: Input data (C, N), where C is the number of classes and N is the number of examples.
    - y: True labels (N,).

    Returns:
    - loss: Scalar cross-entropy loss.
    - dX: Gradient of the loss with respect to X.
    """
    # Calculate unnormalized probabilities (scores)
    exp_scores = np.exp(X - np.max(X, axis=0, keepdims=True))

    # Normalize the scores to get probabilities
    probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    # Calculate the negative log probability of the correct class
    correct_class_log_prob = -np.log(probabilities[y, np.arange(X.shape[1])])

    # Compute the average loss
    loss = np.sum(correct_class_log_prob) / X.shape[1]

    # Compute the gradient of the loss with respect to X
    dX = probabilities.copy()
    dX[y, np.arange(X.shape[1])] -= 1
    dX /= X.shape[1]

    return loss, dX


# Example usage with random data
np.random.seed(42)
C, N = 3, 1  # Number of classes and examples
X = np.random.rand(C, N)
y = np.random.randint(C, size=N)

# Compute softmax cross-entropy loss and gradient
softmax_cross_entropy_loss_value, softmax_cross_entropy_loss_gradient = softmax_cross_entropy_loss(X, y)

print("Softmax Cross-Entropy Loss:", softmax_cross_entropy_loss_value)
print("Softmax Cross-Entropy Loss Gradient:\n", softmax_cross_entropy_loss_gradient)

