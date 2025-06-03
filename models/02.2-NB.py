import torch
import matplotlib.pyplot as plt
import math
from random import random


###
### points, changes, defaults
###

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k = 5
n = 10
m = 20

###
### Prediction function
###

def prediction(x: torch.Tensor, phi: torch.Tensor, p: torch.Tensor):
    """ retun an y-vector prediction to given x, phi and p.

    Given x is (m, n), phi is (n, k) and p is (k, 1), we perform
    NB prediction on the matrices to get an y-vector prediction.
    For k classes, each element in y_i in y is in [0,k)

    
    """
    logPhi = phi.log()      # n, k
    logP = p.log().T        # 1, k

    scores = x @ logPhi + logP
    return scores.argmax(dim=1, keepdim=True)

###
### Generate data points
###

def generateData(k: int, n: int, m: int):
    # generate real P (probability of y matrix)
    p = torch.randint(low=1, high=6, size=(k, 1), device=device).to(torch.float32)
    PSum = p.sum(dim=0)
    p = p / PSum

    # generate real phi
    phi = torch.randint(low=1, high=5, size=(n, k), device=device).to(torch.float32)
    phiSum = phi.sum(dim=0, keepdim=True).expand(size=phi.size())
    phi = phi / phiSum
    phiNot = 1 - phi

    # generate training x
    x = torch.randint(low=0, high=2, size=(m, n), device=device).to(torch.float32)
    xNot = 1 - x

    # generate phi and p, also add temperature
    
    temperature = 0.3

    logPhi = phi.log()
    logPhiNot = phiNot.log()
    logP = p.log().T.expand(size=(m, k))

    # generate training y
    yPercentages = x @ logPhi + xNot @ logPhiNot + logP
    yPercentages = yPercentages / temperature
    y = yPercentages.softmax(dim=1)
    y = y.multinomial(num_samples=1)

    return (x, y, phi, p)

x, y, phiTrue, pTrue = generateData(k, n, m)
    
# generate real P (probability of y matrix)
p = torch.randint(low=1, high=6, size=(k,), device=device)
pSum = p.sum()
p = p / pSum

def accuracy(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, f"a={a.shape}, b={b.shape}"
    return (a == b).float().mean().item() * 100

yPredict = prediction(x, phiTrue, pTrue)

###
### Train, predict params
###

def paramFit(x: torch.Tensor, y: torch.Tensor, classCount: int) -> tuple[torch.Tensor, torch.Tensor]:
    ''' Find phi and p that fits the Naive Bayes model

    Args:
        x           (torch.Tensor): (m, n) tensor, data
        y           (torch.Tensor): (m, 1) tensor, output
        classCount  (int):          amount of classes
    Returns:
        (phi, p) (tuple[torch.Tensor, torch.Tensor]): 
            phi is (n, k) tensor s.t. phi[i][j] = p(x_i | y = j) and
            p is (k, 1) tensor s.t. p[i][0] = p(y = i)
    '''

    assert(x.size()[0] == y.size()[0])
    assert(y.size()[1] == 1)
    
    m = x.size()[0]
    n = x.size()[1]
    k = classCount

    y_onehot = torch.nn.functional.one_hot(y.squeeze(), num_classes=k).to(torch.float32)

    phiNumerator = x.T @ y_onehot + 1
    phiDenominator = y_onehot.sum(dim=0, keepdim=True).expand(size=phiNumerator.shape) + k
    phi = phiNumerator / phiDenominator

    P = y_onehot.mean(dim=0, keepdim=True).T

    return phi, P

###
### Testing
###

phiPredict, pPredict = paramFit(x, y, k)
yPredict = prediction(x, phiPredict, pPredict)

print(f'accuracy: {accuracy(yPredict, y)}')
print(yPredict.T)
print(y.T)