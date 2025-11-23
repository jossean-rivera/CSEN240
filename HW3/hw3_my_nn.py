import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyNN:
    def __init__(self, architecture):
        self.architecture = architecture
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        np.random.seed(42)
        for i in range(len(architecture) - 1):
            self.weights.append(np.random.uniform(low=-1, high=1, size=(architecture[i], architecture[i+1])))
            self.biases.append(np.zeros((1, architecture[i+1])))

    #implementing the relu activation function
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        perceptron_inputs = [X]
        perceptron_outputs = []

        for W, b in zip(self.weights, self.biases):
            ## INSERT  YOUR CODE HERE ###

            # perceptron_inputs will have the activations A0, A1, ..., An
            # perceptron_outputs will have the pre-activations Z1, Z2, ..., Zn

            # Get the current input which is the output of the previous, last layer
            A_last = perceptron_inputs[-1]

            # Calculate the perceptron linear combination
            # z = b + w0*x0 + w1*x1 + ... + wn*xn 
            z = np.dot(A_last, W) + b
            perceptron_outputs.append(z)

            # Apply activation function, ReLU in this case
            a = self.relu(z)
            perceptron_inputs.append(a)
            

        return perceptron_inputs, perceptron_outputs

    def predict(self, X):
        ## INSERT YOUR CODE HERE ###
        perceptron_inputs, _ = self.forward(X)
        return perceptron_inputs[-1].flatten()
    
    def backward(self, perceptron_inputs, perceptron_outputs, target):
        weight_changes = []
        bias_changes = []

        m = len(target)
        # Output layer gradient
        ## INSERT YOUR CODE HERE ### 
        # ReLU activation function: max(0, x)
        # Derivate of ReLU: 1 if x > 0 else 0
        
        # Loss: Mean Squared Error loss: L = (1/m) * sum((y - t)^2)
        # Need to calculate gradients for weights dL/dW and biases dL/db
        # dL/dW = dL/dy * dy/dz * dz/dW
        # dL/db = dL/dy * dy/dz * dz/db

        # Derivate of Loss: dL/dy = (2/m) * (y - t)
        y = perceptron_inputs[-1]
        dL_dy = (2/m) * (y - target.reshape(-1, 1))

        for i in reversed(range(len(self.weights))):
            ## INSERT YOUR CODE HERE ###
            
            # Get pre-activation (z) and 
            # activation output (y) of the current layer
            z = perceptron_outputs[i]
            y = perceptron_inputs[i]

            # Calculate derivative dL/dz = dL/dy * dy/dz = dL/dy * ReLU'(z)
            dL_dz = dL_dy * self.relu_derivative(z)

            # Calculate gradients for weights and biases
            # dL/dW = dL/dz * dz/dW = dL/dz * y_prev
            dL_dW = np.dot(y.T, dL_dz)
            # dL/db = dL/dz * dz/db = dL/dz * 1
            dL_db = np.sum(dL_dz, axis=0, keepdims=True)

            # Update weights and biases for next layer
            weight_changes.append(dL_dW)
            bias_changes.append(dL_db)

            # Prepare dL/dy for the next layer
            dL_dy = np.dot(dL_dz, self.weights[i].T)

        return list(reversed(weight_changes)), list(reversed(bias_changes))
    
    def update_weights(self, weight_changes, bias_changes, lr):
        for i in range(len(self.weights)):
            ## INSERT YOUR CODE HERE ##
            # Apply gradient descent update
            # w_new = w_old - lr * dL/dW
            self.weights[i] = self.weights[i] - lr * weight_changes[i]
            self.biases[i] = self.biases[i] - lr * bias_changes[i]

    def train(self, X, y, epochs, lr=0.01):
        for epoch in range(epochs):
            perceptron_inputs, perceptron_outputs = self.forward(X)
            weight_changes, bias_changes = self.backward(perceptron_inputs, perceptron_outputs, y)
            self.update_weights(weight_changes, bias_changes, lr)

            if epoch % 20 == 0 or epoch == epochs - 1:
                loss = np.mean((perceptron_inputs[-1].flatten() - y) ** 2)  # MSE
                print(f"EPOCH {epoch}: Loss = {loss:.4f}")
    

# Define a random function with two inputs
def random_function(x, y):
    # Function: f(x, y) = sin(x) + x*cos(y) + y + 3^(x/3)
    return (np.sin(x) + x * np.cos(y) + y + 3**(x/3))

# Define the number of random samples to generate
n_samples = 1000

# Generate random X and Y values within a specified range
x_min, x_max = -10, 10
y_min, y_max = -10, 10

# Generate random values for X and Y
X_random = np.random.uniform(x_min, x_max, n_samples)
Y_random = np.random.uniform(y_min, y_max, n_samples)

# Evaluate the random function at the generated X and Y values
Z_random = random_function(X_random, Y_random)

# Create a dataset
dataset = pd.DataFrame({
    'X': X_random,
    'Y': Y_random,
    'Z': Z_random
})

# Display the dataset
print(dataset.head())

skip = True

if (skip != True):
    # Create a 2D scatter plot of the sampled data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(dataset['X'], dataset['Y'], c=dataset['Z'], cmap='viridis', s=10)
    plt.colorbar(scatter, label='Function Value')
    plt.title('Scatter Plot of Randomly Sampled Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.show(block=False)
    plt.pause(5)
    plt.close()

# Flatten the data
X_flat = X_random.flatten()
Y_flat = Y_random.flatten()
Z_flat = Z_random.flatten()

# Stack X and Y as input features
inputs = np.column_stack((X_flat, Y_flat))
outputs = Z_flat

# Normalize the inputs and outputs
inputs_mean = np.mean(inputs, axis=0)
inputs_std = np.std(inputs, axis=0)
outputs_mean = np.mean(outputs)
outputs_std = np.std(outputs)

inputs = (inputs - inputs_mean) / inputs_std
outputs = (outputs - outputs_mean) / outputs_std

# Define the architecture: [input_dim, hidden1, ..., output_dim]
## FEEL FREE TO MODIFY THE ARCHITECTURE ###
architecture = [2, 16, 16, 16, 16, 1]  # Two inputs, ..., one output
model = MyNN(architecture)

## FEEL FREE TO MODIFY N_EPOCHS, LR ###
N_EPOCHS = 500
LR = 0.0001
# Train the model
model.train(inputs, outputs, epochs=N_EPOCHS, lr=LR)

# Reshape predictions to grid format 
Z_pred = model.predict(inputs) * outputs_std + outputs_mean
Z_pred = Z_pred.reshape(X_random.shape)


mse = ((Z_pred - Z_flat)**2).mean(axis=0)
print(f"Test MSE: {mse:>6f} \n")

plt.plot(Z_pred, c= 'blue')
plt.plot(Z_flat, c = 'red')
plt.show(block=False)
plt.pause(25)
plt.close()