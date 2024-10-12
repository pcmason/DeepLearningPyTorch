"""
This is a copy of the program from pytorchMLP.py, but is updated to keep track of the training loop accuracy and
accuracy of the partially trained model and output using matplotlib. Also updated to use tqdm to show progress bar
while training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm

# Load Pimas Indians Diabetes dataset
dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Now make sure x and y variables are represented as tensors for PyTorch
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Split dataset into training and test sets
xTrain = x[:700]
yTrain = y[:700]
xTest = x[700:]
yTest = y[700:]

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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 50
batch_size = 10
batches_per_epoch = len(xTrain)

# Collect statistics
train_loss, train_acc, test_acc = [], [], []

starts = [i * batch_size for i in range(batches_per_epoch)]

# Nested for loop going through epochs and batches to train model
for epoch in range(n_epochs):
    with tqdm.tqdm(starts, unit='batch', mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # Take a batch
            xbatch = xTrain[start: start + batch_size]
            ybatch = yTrain[start: start + batch_size]
            # Forward pass
            y_pred = model(xbatch)
            loss = loss_fn(y_pred, ybatch)
            acc = (y_pred.round() == ybatch).float().mean()
            # Store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
            # Print progress
            bar.set_postfix(
                loss=float(loss),
                acc=f"{float(acc)*100:.2f}%"
            )
    # Evaluate model at end of epoch
    y_pred = model(xTest)
    acc = (y_pred.round() == yTest).float().mean()
    test_acc.append(float(acc))
    print(f"End of {epoch} accuracy {acc}")

# Plot the accuracy metrics
avg_train_acc = []
# Need to average out the training set metrics to work with the epoch-level test metric
for i in range(n_epochs):
    start = i * batch_size
    average = sum(train_acc[start:start+batches_per_epoch]) / batches_per_epoch
    avg_train_acc.append(float(average))
plt.plot(avg_train_acc, label='train')
plt.plot(test_acc, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0)
plt.show()

# Plot the loss metrics
plt.plot(train_loss)
plt.xlabel('steps')
plt.ylabel('loss')
plt.ylim(0)
plt.show()

# Compute accuracy, no_grad is suggested as it relieves y_pred from remembering how it comes up with answer
with torch.no_grad():
    y_pred = model(xTest)
accuracy = (y_pred.round() == yTest).float().mean()
print(f"Accuracy {accuracy}")
