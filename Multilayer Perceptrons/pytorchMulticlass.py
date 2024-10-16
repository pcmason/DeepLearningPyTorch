"""
This program is a multiclass classification pytorch model to solve the iris flower classification problem. Model uses
benchmarking and plots the cross entropy and accuracy as the model trains at the end.
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Read data and apply one-hot encoding
data = pd.read_csv('iris.csv', header=None)
x = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
y = ohe.transform(y)

# Convert pandas dataframe and numpy array into PyTorch tensors
x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)



# Create the multiclass model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

# Loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare model and training parameters
n_epochs = 200
batch_size = 5
batches_per_epoch = len(x_train)

best_acc = -np.inf  # init to negative infinity
best_weights = None
train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = [], [], [], []

# Training loop
for epoch in range(n_epochs):
    epoch_loss, epoch_acc = [], []
    # Set model in training mode and run through each batch
    model.train()
    with tqdm.trange(batches_per_epoch, unit='batch', mininterval=0) as bar:
        bar.set_description(f'Epoch {epoch}')
        for i in bar:
            # Take a batch
            start = i * batch_size
            x_batch = x_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # Forward pass
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
            # Compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            # Issue where there are a lot of nan being computed, only append real values to epoch_loss
            if loss.item() != 'nan':
                epoch_loss.append(float(loss))
            if acc.item() != 'nan':
                epoch_acc.append(float(acc))
            bar.set_postfix(loss=float(loss), acc=float(acc))
    # Set model in evaluation mode and go through test set
    model.eval()
    y_pred = model(x_test)
    ce = float(loss_fn(y_pred, y_test))
    acc = float((torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean())
    # Do this to remove all of the nan, as the above method did not fully work
    epoch_loss = [x for x in epoch_loss if x == x]
    epoch_acc = [x for x in epoch_acc if x == x]
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%')

# Restore best model
model.load_state_dict(best_weights)

print(train_loss_hist)
print(test_loss_hist)

# Plot loss
plt.plot(train_loss_hist, label='train')
plt.plot(test_loss_hist, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(train_acc_hist, label='train')
plt.plot(test_acc_hist, label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
