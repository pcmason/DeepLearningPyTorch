"""
This is a simple binary classification PyTorch MLP model that is created and evaluated on the Pimas Indians Diabetes
dataset. Model predicts whether individual has early onset diabetes or not based on the input data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load Pimas Indians Diabetes dataset
dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Now make sure x and y variables are represented as tensors for PyTorch
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# Train the model
loss_fn = nn.BCELoss()  # Binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

# Nested for loop going through epochs and batches to train model
for epoch in range(n_epochs):
    for i in range(0, len(x), batch_size):
        xbatch = x[i: i + batch_size]
        y_pred = model(xbatch)
        ybatch = y[i: i + batch_size]
        # Calculate the loss and update the network
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Finished epoch {epoch}, latest loss {loss}")

# Compute accuracy, no_grad is suggested as it relieves y_pred from remembering how it comes up with answer
with torch.no_grad():
    y_pred = model(x)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# Make class predictions with the model
predictions = (model(x) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
