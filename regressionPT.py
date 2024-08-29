"""
Simple linear regression model from PyTorch trained for 2 parameters (w & b) to attempt to learn a dataset that is
generated in this program
"""
import torch
import matplotlib.pyplot as plt


# Create method to create dataset
def create_data():
    x = torch.arange(-5, 5, 0.1).view(-1, 1)
    func = -5 * x
    y = func + 0.4 * torch.randn(x.size())
    return x, y


# Define method to forward pass for prediction
def forward(x, w, b):
    return w * x + b


# Evaluating data points with MSE
def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)


# Create the main method
def main():
    # Load in data
    x, y = create_data()
    # Create the w and b parameters
    w = torch.tensor(-10.0, requires_grad=True)
    b = torch.tensor(-20.0, requires_grad=True)

    step_size = 0.1
    loss_list = []
    iter = 20

    for i in range(iter):
        # Make predictions with forward pass
        y_pred = forward(x, w, b)
        # Calculate loss between original and predicted data
        loss = criterion(y_pred, y)
        # Store calculated loss in list
        loss_list.append(loss.item())
        # Backward pass for computing gradients of the loss WRT learnable parameters
        loss.backward()
        # Update parameters after each iteration
        w.data = w.data - step_size * w.grad.data
        b.data = b.data - step_size * b.grad.data
        # Zeroing gradients after each iteration
        w.grad.data.zero_()
        b.grad.data.zero_()
        # Printing the values for understanding
        print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))

    # Plot the loss after each iteration
    plt.plot(loss_list, 'r')
    #plt.tight_layout()
    plt.grid('True', color='y')
    plt.xlabel('Epochs/Iterations')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    main()