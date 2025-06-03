import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random

#
#   points, changes, defaults
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thetaOriginal = [70, 108, -16, 42, -20, -55]
m = 1000
n = len(thetaOriginal)
alpha = 5e-4

#
#   Generate data points
#


xData = []
yData = []

averageDeviation = 50 # deviation for non-perfect y
def linearFunction(x: list, theta: list) -> float:
    assert(len(x) == len(theta))
    s = sum(x[i] * theta[i] for i in range(len(x)))
    return s

xData = []
yData = []

for i in range(m):
    xNew = []
    for k in range(n - 1):
        xNew.append(random() * 50)
    xNew.append(1)

    xData.append(xNew)
    yData.append([
        linearFunction(xNew, thetaOriginal) + random() * 2 * averageDeviation - averageDeviation])
    
#
#   Convert x, y to torch matrices
#

# x:        R(m, n)
# y:        R(m, 1)
# theta:    R(n, 1)

x = torch.tensor(xData, dtype=torch.float32, device=device)
y = torch.tensor(yData, dtype=torch.float32, device=device)
theta = torch.zeros((n, 1), dtype=torch.float32, device=device)

###
#   Update rule
###

def updateThetaOneEpoch(x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, alpha: float) -> torch.Tensor:
    m = y.shape[0]
    pred = x @ theta
    error = pred - y
    grad = x.T @ error
    return theta - alpha / m * grad

###
#   Testing
###

# --- Training loop (no graphs) ---
epochs = 100000
for epoch in range(epochs):
    pred = x @ theta
    loss = ((pred - y) ** 2).mean().item()

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | MSE Loss: {loss:.4f}")

    theta = updateThetaOneEpoch(x, y, theta, alpha)

# --- Final results ---
print("\nFinal learned theta:")
print(theta.flatten().tolist())

print("\nGround truth theta:")
print(thetaOriginal)