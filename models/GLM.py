import torch
import matplotlib.pyplot as plt
import math
from random import random

###
### points, changes, defaults
###

k = 43      # classification amount
n = 50     # input feature count
m = 500    # training set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###
### Generate data points
###

x = torch.rand(size=(m, n), dtype=torch.float32, device=device)
thetaTrue = torch.rand(size=(n, k), dtype = torch.float32, device=device)

eta = x @ thetaTrue

# probability, essentially true hypothesis
probs = torch.softmax(eta, dim=1, dtype=torch.float32)
yChoice = torch.multinomial(probs, num_samples=1)
y = torch.nn.functional.one_hot(yChoice.squeeze(1), num_classes=k).float()

theta = torch.zeros(size=(n, k), dtype=torch.float32, device=device)
    
###
### Some g(z) canonical link functions
###

def softmax(matrix: torch.Tensor) -> torch.Tensor:
    return torch.softmax(matrix, dim=1, dtype=torch.float32)

def linear(matrix: torch.Tensor) -> torch.Tensor:
    return matrix

def logistic(matrix: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(matrix)

###
### Update rule
###

def glmUpdate(x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, alpha: float, link) -> torch.Tensor:
    m = y.shape[0]
    eta = x @ theta
    hypothesis = link(eta)
    error = y - hypothesis
    grad = x.T @ error
    return theta + alpha / m * grad

###
### Testing
###

# Settings
alpha = 1
iterations = 50000
loss_history = []

# Cross-entropy loss for checking progress
def cross_entropy(y_true, y_pred):
    return -(y_true * torch.log(y_pred + 1e-9)).sum(dim=1).mean()

# Training loop
for i in range(iterations):
    theta = glmUpdate(x, y, theta, alpha, softmax)
    preds = softmax(x @ theta)
    loss = cross_entropy(y, preds)
    loss_history.append(loss.item())

    if i % 500 == 0:
        print(f"Epoch {i:4d} | Loss: {loss.item():.4f}")

# Plot loss
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.savefig('pltPlots/01.3-softmax.png')
