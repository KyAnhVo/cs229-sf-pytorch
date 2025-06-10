import torch
import matplotlib.pyplot as plt
import math
from random import random


###
### points, changes, defaults
###

m = 100
n = 40
k = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###
### Generate data points
###

mu = torch.randn((k, n), device=device)           # means for each class
L = torch.eye(n, device=device) * 0.1             # Cholesky of shared cov
y = torch.randint(0, k, (m,), device=device)      # true class labels
x = mu[y] + torch.randn((m, n), device=device) @ L.T  # draw from N(mu_y, Sigma)
y = y.unsqueeze(1)
    
###
### Train, predict params
###

# probability and y_onehot matrices
y_onehot = torch.nn.functional.one_hot(y.squeeze(1), num_classes=k).float()
p = torch.mean(y_onehot, dim=0).unsqueeze(1)
p = p.clamp(min=1e-8)
print(f'y_onehot.shape = (m, k) is {y_onehot.shape == (m, k)}')
print(f'p.shape = (k, 1) is {p.shape == (k, 1)}')
logp = torch.log(p)

# mean matrix
numerator = y_onehot.T @ x
denominator = m * p
mean = numerator / denominator
print(f'mean.shape = (k, n) is {mean.shape == (k, n)}')

# calculating sigma
mean_y = mean[y.squeeze()]
diff = x - mean_y
diff_unsqueezed = diff.unsqueeze(2)
outer = diff_unsqueezed @ diff_unsqueezed.transpose(1, 2)
sigma = outer.mean(dim=0)
det = torch.det(sigma)

###
### Predict function
###

def predict(x:torch.Tensor):
    assert(x.shape == (1, n))
    diff = x.expand(mean.shape) - mean
    temp = diff @ sigma.inverse() # (k, n) * (n, n) = (k, n)
    scores = (diff * temp).sum(dim=1) # (k, 1)
    scores = -0.5 * scores + logp.squeeze()
    
    return scores.argmax()

###
### Testing
###

testAmount = 50

# Pick 30 test indices (e.g., first testAmount, or random)
test_indices = torch.randperm(m, device=device)[:testAmount]
x_test = x[test_indices]        # shape: (testAmount, n)
y_test = y[test_indices].squeeze()  # shape: (testAmount,)

# Predict each sample (loop-based, since predict expects shape (1, n))
y_pred = torch.tensor([
    predict(xi.unsqueeze(0)).item() for xi in x_test
], device=device)

# Compute accuracy
correct = (y_pred == y_test).sum().item()
total = len(y_test)
accuracy = correct / total

print(f"Accuracy on {testAmount} test samples: {accuracy:.2%} ({correct}/{total})")
print(f"sigma condition number: {torch.linalg.cond(sigma).item():.2e}")
print("Class counts:", y.squeeze().bincount(minlength=k).tolist())
