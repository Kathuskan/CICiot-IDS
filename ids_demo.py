
import numpy as np
import torch
from torch import nn

# 1. Make results reproducible
np.random.seed(42)
torch.manual_seed(42)

# 2. Generate an illustrative dataset
# Each row represents one network flow.
n = 2000
y = np.random.randint(0, 2, size=n)

# Four features:
# duration, packet_count, bytes_per_second, failed_connection_rate
duration = np.where(
    y == 0,
    np.random.lognormal(0.7, 0.7, n),
    np.random.lognormal(-0.1, 0.8, n)
)

packet_count = np.where(
    y == 0,
    np.random.poisson(25, n) + 1,
    np.random.poisson(65, n) + 1
)

bytes_per_second = np.where(
    y == 0,
    np.random.lognormal(7.5, 0.8, n),
    np.random.lognormal(8.5, 0.9, n)
)

failed_connection_rate = np.where(
    y == 0,
    np.random.beta(1, 12, n),
    np.random.beta(3, 5, n)
)

X = np.column_stack([
    duration,
    packet_count,
    bytes_per_second,
    failed_connection_rate
]).astype(np.float32)

y = y.astype(np.float32)

# 3. Split into training and testing sets
rng = np.random.default_rng(42)
indices = rng.permutation(n)

split = int(0.8 * n)
train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# 4. Standardize features using training data only
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 5. Convert NumPy arrays into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

# 6. Define the neural network
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 7. Define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 8. Train the model
epochs = 100

for epoch in range(epochs):
    model.train()

    # Forward pass
    logits = model(X_train)

    # Calculate prediction error
    loss = loss_fn(logits, y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )

# 9. Evaluate on data the model did not train on
model.eval()

with torch.no_grad():
    test_logits = model(X_test)
    probabilities = torch.sigmoid(test_logits)
    predictions = (probabilities >= 0.5).float()

    accuracy = (predictions == y_test).float().mean()

print(f"\nTest accuracy: {accuracy.item():.2%}")

# 10. Predict one new illustrative flow
new_flow = np.array([[
    0.5,       # duration in seconds
    75,        # packet count
    6000,      # bytes per second
    0.35       # failed connection rate
]], dtype=np.float32)

new_flow = (new_flow - mean) / std
new_flow = torch.tensor(new_flow, dtype=torch.float32)

with torch.no_grad():
    score = torch.sigmoid(model(new_flow)).item()

print(f"Suspicious-traffic score: {score:.3f}")
print(
    "Prediction:",
    "Suspicious" if score >= 0.5 else "Normal"
)
