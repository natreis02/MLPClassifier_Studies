'''
Script to create 1000 images (2x2) and train/test using Multilayer-Perceptron Classifier

Authors: Natália França dos Reis and Vitor Hugo Miranda Mourão
date: 08/19/2023
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set a random seed for reproducibility
np.random.seed(42)  # You can use any seed value you prefer

# Number of data points
num_data_points = 1000

# Initialize an empty array to store the data
data = []

# Generate the data
for i in range(num_data_points):
    # Create a 2x2 random pixel matrix with values between 0 and 255
    pixel_matrix = np.random.randint(0, 256, size=(2, 2))
    data.append(pixel_matrix)
    
    
# Convert the list of pixel matrices to a numpy array
data = np.array(data)

'''
# Plot the first 9 images
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(data[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"Image {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()
'''

labels = []

for pixel_matrix in data:
    min_pixel_value = np.min(pixel_matrix)
    min_pixel_position = np.argwhere(pixel_matrix == min_pixel_value)[0]
    
    if min_pixel_position[0] == 0 and min_pixel_position[1] == 0:
        labels.append('A')
    elif min_pixel_position[0] == 0 and min_pixel_position[1] == 1:
        labels.append('B')
    elif min_pixel_position[0] == 1 and min_pixel_position[1] == 0:
        labels.append('C')
    else:
        labels.append('D')
        
# Convert the list of labels to a numpy array
labels = np.array(labels)
        
# Split the data and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Flatten the pixel matrices for input to the MLP
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Create an MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(3), activation= "relu", solver = "adam", max_iter=5000, random_state=42)

# Train the MLP on the training data
mlp_classifier.fit(X_train_flatten, y_train)

# Predict on the test data
y_pred = mlp_classifier.predict(X_test_flatten)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Find indices of failed predictions
failed_indices = np.where(y_pred != y_test)[0]

# Plot the first four failed predictions
fig, axes = plt.subplots(1, 4, figsize=(10, 5))

for i, ax in enumerate(axes):
    idx = failed_indices[i]
    ax.imshow(X_test[idx], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"True: {y_test[idx]}, Predicted: {y_pred[idx]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

print("Failed indices: ", failed_indices, "\n")

# Access the weights and biases
weights = mlp_classifier.coefs_
biases = mlp_classifier.intercepts_

# Print the weights and biases for each layer
for i, (weight, bias) in enumerate(zip(weights, biases)):
    print(f"Layer {i + 1} - Weights:\n{weight}\nBiases:\n{bias}\n")

# Create new matrix
new_matrix = np.array([[100, 150], [100, 200]])  

# Resize matrix
new_matrix_flatten = new_matrix.reshape(1, -1)

# Prevision class new matrix
predicted_class = mlp_classifier.predict(new_matrix_flatten)[0]

# Decode matrix in (A, B, C ou D)
classes = ['A', 'B', 'C', 'D']
print("Predicted class for the new matrix:", predicted_class)
