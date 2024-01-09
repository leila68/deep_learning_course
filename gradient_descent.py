import numpy as np
import matplotlib.pyplot as plt

def quadratic_objective(x, y, a, b, c):
    return a * x**2 + b * y**2 + c * x * y

def gradient(x, y, a, b, c):
    grad_x = 2 * a * x + c * y
    grad_y = 2 * b * y + c * x
    return np.array([grad_x, grad_y])

def stochastic_gradient_descent(a, b, c , learning_rate=0.01, num_iterations=100):
    x = np.random.randn()
    y = np.random.randn()
    history = []

    for i in range(num_iterations):
        current_loss = quadratic_objective(x, y, a, b, c)
        history.append((x, y, current_loss))

        grad = gradient(x, y, a, b, c)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]

    return x, y, history

def main():
    # Coefficients of the quadratic objective function
    a = 2
    b = 3
    c = 1

    # Stochastic Gradient Descent
    final_x, final_y, history = stochastic_gradient_descent(a, b, c)

    # Print the final result
    print(f'Final x: {final_x}, Final y: {final_y}, Final Loss: {quadratic_objective(final_x, final_y, a, b, c)}')

    # Plot the optimization path
    x_values, y_values, losses = zip(*history)
    plt.plot(losses)
    plt.title('Optimization Path')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.show()

if __name__ == "__main__":
    main()

