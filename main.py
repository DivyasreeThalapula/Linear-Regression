import numpy as np

# 1. Data
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([5, 7, 9, 11, 13], dtype=float)

# 2. Initialize parameters
w = 0.0
b = 0.0

# 3. Initialize learning rate
learning_rate = 0.01

# 3. Training
for epoch in range(1000):

    # Forward Pass
    y_pred = w * X + b

    # Loss
    loss = np.mean((y_pred - Y) ** 2)

    # Backpropagation
    error = y_pred - Y

    grad_w = np.mean(2 * X * error)
    grad_b = np.mean(2 * error)

    # Gradient Descent
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # Print progress
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {loss:.6f} | "
            f"w: {w:.6f} | "
            f"b: {b:.6f}"
        )


# 4. Test
print("\nFinal parameters:")
print("w =", w)
print("b =", b)

x_new = float(input("\nEnter a value for x: "))

prediction = w * x_new + b

print("\nPrediction:")
print("For x =", x_new)
print("Predicted y =", prediction)