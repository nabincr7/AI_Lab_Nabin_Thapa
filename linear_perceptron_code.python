import numpy as np

# Define the Perceptron class
class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.rand(num_inputs + 1)  # +1 for bias
        self.learning_rate = learning_rate

    def linear(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]  # Weighted sum

    def predict(self, inputs):
        return self.linear(inputs)  # No activation function for linear output

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = prediction - target
        self.weights[1:] -= self.learning_rate * error * inputs  # Update weights
        self.weights[0] -= self.learning_rate * error  # Update bias

    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            mse = 0
            for inputs, target in zip(X, y):
                self.train(inputs, target)
                mse += (self.predict(inputs) - target) ** 2
            mse /= len(y)  # Mean Squared Error
            print(f'Epoch [{epoch + 1}/{num_epochs}], MSE: {mse:.4f}, Weights: {self.weights[1:]}, Bias: {self.weights[0]:.4f}')

# Generate random input data
np.random.seed(42)  # For reproducibility
X = np.random.rand(10, 3)  # 10 samples, 3 features
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5  # Compute target output

# Initialize and train the Perceptron
perceptron = Perceptron(num_inputs=3)
perceptron.fit(X, y, num_epochs=100)

# Final learned weights and bias
print(f'Final Weights: {perceptron.weights[1:]}, Final Bias: {perceptron.weights[0]:.4f}')
