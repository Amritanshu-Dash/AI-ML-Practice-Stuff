import numpy as np

class NeuralNetLayer:
    def __init__(self, input_size, output_size, seed=42):
        np.random.seed(seed)
        #weight matrix (output_size x input_size)
        self.weights = np.random.randn(output_size, input_size) * 0.01
        #bias vector (output_size x 1)
        self.biases = np.zeros((output_size, 1))

    def forward(self, input_data):
        """
        Perform the forward pass through the layer.

        Parameters:
        input_data (numpy.ndarray): Input data of shape (input_size, number_of_samples or batch_size)

        Returns:
        numpy.ndarray: Output data of shape (output_size, number_of_samples)
        """
        self.input_data = input_data
        # Linear transformation: Z = W.X + b
        output_data = self.weights @ input_data + self.biases
        return output_data

    def relu(self, output_data):
        """Apply ReLU activation function."""
        self.output_data = output_data
        return np.maximum(0, output_data)

    def relu_derivative(self, dA):
        """Compute the derivative of ReLU activation."""
        return dA * (self.output_data > 0)

    def backward(self, dA, learning_rate=0.01):
        """
        Perform the backward pass through the layer.

        Parameters:
        dA (numpy.ndarray): Gradient of the loss with respect to the output of this layer.
        learning_rate (float): Learning rate for weight updates.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input of this layer.

        dA = upstream gradient (from next layer)
        """
        batch_size = dA.shape[1]
        dZ = self.relu_derivative(dA)  # Gradient through ReLU
        dw = (dZ @ self.input_data.T) / batch_size  # Gradient w.r.t weights
        db = np.sum(dZ, axis=1, keepdims=True)/ batch_size  # Gradient w.r.t biases
        dX = self.weights.T @ dZ  # Gradient w.r.t input data

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        return dX


layer = NeuralNetLayer(input_size=3, output_size=2)

# Example input (3 features, 4 samples)
X = np.array([[1, 2, 3, 4],
              [0, 1, 0, 1],
              [2, 0, 1, 0]]).astype(float)
# Forward pass
Z = layer.forward(X)
A = layer.relu(Z)
print('input to layer X:\n', X)
print('output from layer A after ReLU:\n', A)

# Example upstream gradient (from next layer)
dA = np.array([[1.0, -1.0, 0.5, -0.5],
               [0.5, 0.5, 1.0, -1.0]]).astype(float)
# Backward pass
dx = layer.backward(dA, learning_rate=0.01)

print('Weights before update:\n', layer.weights)
layer.forward(X)  # Forward again to see updated weights effect
print('Weights after update:\n', layer.weights)
