"""
Create a simple MLP using PyTorch to demonstrate binary classification using the Ionosphere binary classification
dataset. This is a simple example that is well commented to be used as a guide for future PyTorch models
"""
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD


# Dataset definition
class CSVDataset(Dataset):
    # Load dataset
    def __init__(self, path):
        # Load csv as dataframe
        df = read_csv(path, header=None)
        # Store inputs and outputs
        self.x = df.values[:, :-1]
        self.y = df.values[:, -1]
        # Ensure input data is floats
        self.x = self.x.astype('float32')
        # Label encode target and ensure values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # Number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # Get a row at an index
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

    # Get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # Determine sizes
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        # Calculate the split
        return random_split(self, [train_size, test_size])


# Model definition
class MLP(Module):
    # Define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # Input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # Second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # Third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # Forward propagate input
    def forward(self, x):
        # Input to first hidden layer
        x = self.hidden1(x)
        x = self.act1(x)
        # Second hidden layer
        x = self.hidden2(x)
        x = self.act2(x)
        # Third hidden layer and output
        x = self.hidden3(x)
        x = self.act3(x)
        return x


# Prepare the dataset
def prepare_data(path):
    # Load dataset
    dataset = CSVDataset(path)
    # Calculate split
    train, test = dataset.get_splits()
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# Train the model
def train_model(train_dl, model):
    # Define optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Enumerate epochs
    for epoch in range(100):
        # Enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # Clear gradients
            optimizer.zero_grad()
            # Compute the model output
            yhat = model(inputs)
            # Calculate loss
            loss = criterion(yhat, targets)
            # Credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# Evaluate model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # Evaluate model on test set
        yhat = model(inputs)
        # Retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # Round to class values
        yhat = yhat.round()
        # Store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # Calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# Make a class prediction for one row of data
def predict(row, model):
    # Convert row to data
    row = Tensor([row])
    # Make prediction
    yhat = model(row)
    # Retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# Use main method to prepare and run model
def main():
    path = 'ionosphere.csv'
    train_dl, test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    # Define the network
    model = MLP(34)
    # Train the model
    train_model(train_dl, model)
    # Evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)
    # Make a single prediction
    row = [1, 0, 0.99539, -0.05889, 0.85243, 0.02306, 0.83398, -0.37708, 1, 0.03760, 0.85243, -0.17755, 0.59755,
           -0.44945, 0.60536, -0.38223, 0.84356, -0.38542, 0.58212, -0.32192, 0.56971, -0.29674, 0.36946, -0.47357,
           0.56811, -0.51171, 0.41078, -0.46168, 0.21266, -0.34090, 0.42267, -0.54487, 0.18641, -0.45300]
    yhat = predict(row, model)
    print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))


if __name__ == '__main__':
    main()
