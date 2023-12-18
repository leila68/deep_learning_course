import numpy as np

def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  # see notes about delta later in this section
  delta = 1.0
  # scores becomes of size 10 x 1, the scores for each class
  scores = W.dot(x)
  correct_class_score = scores[y]
  # number of classes, e.g. 10
  D = W.shape[0]
  loss_i = 0.0
  # iterate over all wrong classes
  for j in range(D):
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i


def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  delta = 1.0
  scores = W.dot(X)
  # Create a mask for the correct class scores
  correct_class_scores = scores[y, np.arange(X.shape[1])]

  # Compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - correct_class_scores + delta)

  # On y-th position, scores[y] - correct_class_scores[y] canceled and gave delta.
  # We want to ignore the y-th position and only consider the margin on the max wrong class
  margins[y, np.arange(X.shape[1])] = 0

  # Compute the loss for the entire dataset
  # loss = np.sum(margins) / X.shape[1]

  loss = np.sum(margins, axis=0)

  return loss

# Example usage with random data
# Define a random image vector (excluding the bias term)
x_without_bias = np.random.rand(3072, 1)

# Append the bias term in the last position
bias = 1
x = np.vstack([x_without_bias, [bias]])

y = np.random.randint(10)
num_classes = 10
num_features = 3073
W = np.random.rand(num_classes, num_features)

# Compute the loss using both implementations
svm_unvec_loss = L_i(x, y, W)
svm_half_vec_loss = L_i_vectorized(x, y, W)

print("Unvectorized SVM Loss:", svm_unvec_loss)
print("Half-Vectorized SVM Loss:", svm_half_vec_loss)


# Example usage with random data
np.random.seed(42)
D, N, C = 3073, 50000, 10  # Number of features, examples, and classes
X1 = np.random.rand(D, N)
y1 = np.random.randint(C, size=N)


svm_full_vec_loss = L(X1, y1, W)

print("Full-Vectorized SVM Loss ", svm_full_vec_loss)
