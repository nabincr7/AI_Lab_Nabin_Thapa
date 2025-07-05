import numpy as np

np.random.seed(24)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, epochs=100):
        mse_history = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict(X)
            # Compute mean squared error
            mse = np.mean((y_pred - y) ** 2)
            mse_history.append(mse)
            print(f"Epoch {epoch + 1}, MSE: {mse:.6f}")

            # Compute gradients
            error = y_pred - y
            dw = np.dot(X.T, error) / len(y)
            db = np.mean(error)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return mse_history

def generate_dataset(n_features, n_samples=10, bias=5):
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    target_weights = np.random.uniform(-1, 1, n_features)
    y = np.dot(X, target_weights) + bias
    return X, y, target_weights, bias

def run_perceptron(n_features):
    print(f"\n=== Perceptron with {n_features} Features ===")
    # Generate dataset
    X, y, target_weights, target_bias = generate_dataset(n_features)
    print("Target Weights:", target_weights)
    print("Target Bias:", target_bias)

    # Initialize and train Perceptron
    perceptron = Perceptron(input_size=n_features, learning_rate=0.01)
    perceptron.train(X, y, epochs=100)

    # Print final weights and bias
    print("\nFinal Weights:", perceptron.weights)
    print("Final Bias:", perceptron.bias)

# Test with n=4 and n=5 features
run_perceptron(4)
run_perceptron(5)