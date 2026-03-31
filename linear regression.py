import numpy as np
import matplotlib.pyplot as plt

# Dataset: House size (sq ft) vs Price
x = np.array([500, 800, 1200, 1500, 1800])
y = np.array([100, 150, 200, 250, 300])

n = len(x)

# Initialize parameters
m = 0
b = 0

learning_rate = 0.0000001
epochs = 200

loss_history = []

for epoch in range(epochs):

    # Step 1: Predict values
    y_pred = m * x + b

    # Step 2: Calculate Mean Squared Error loss
    loss = np.mean((y - y_pred) ** 2)
    loss_history.append(loss)

    # Step 3: Compute gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # Step 4: Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Print final learned equation
print("Final Learned Equation:")
print("Price =", m, "* Size +", b)

# Plot Loss vs Epoch
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.show()