import torch
from typing import Callable

# boilerplate stuffs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 300
m = 5000

###
### Kernel functions
###

def predictionGaussianKernel(xTraining: torch.Tensor, xInput: torch.Tensor, variance: float) -> torch.Tensor:
    ''' returns the Gaussian (RBF) Kernel of the given column vectors, (m, n) x (n, 1)
    
    Args:
        xTraining (torch.Tensor):  training data (m, n)
        xInput (torch.Tensor):  second column vector (n, 1)
        variance (float):   variance of Kernel
    Returns:
        dot product of the two kernelized vectors of x1 and x2.

    '''
    _, n = xTraining.size()
    assert(xInput.size() == (n, 1)), f'input mismatch: {xTraining.size()}, {xInput.size()}'
    
    # Property: |x - y|^2 = |x|^2 + |y|^2 - 2<x, y>
    xInputL2 = (xInput.T ** 2).sum(dim=1, keepdim=True).expand(size=(m, 1))
    xTrainingL2 = (xTraining ** 2).sum(dim=1, keepdim=True)
    xDotX = xTraining @ xInput
    K = torch.exp(-(xInputL2 + xTrainingL2 - xDotX) / (2 * variance))
    return K



def trainingGaussianKernel(x: torch.Tensor, variance: float) -> torch.Tensor:
    ''' returns the training gaussian kernel matrix on matrix x, (m, n)

    Args:
        x (torch.Tensor): (m, n) matrix
        variance (float): variance of the Gaussian kernel / distribution
    Returns:
        (m, m) tensor K s.t. K[i][j] = K(x(i), x(j))
    '''
    assert(x.dim() == 2), "x is not a matrix"
    m, _ = x.size()

    # for each training set (row), we caclulate its l2 sqared length
    xLen = (x ** 2).sum(dim= 1) # (m,) 
    
    # We use the property |x - y|^2 = |x|^2 + |y|^2 - 2<x, y>
    # Applying to matrix, we havexLen
    # xLen(expressed s.t. every row is the same) + xLen(expressed s.t. every column is the same) + x @ X.T (m, m)
    # So each element [i][j] is |x(i) - x(j)|^2
    kernel = xLen.unsqueeze(1).expand(size=(m, m)) + xLen.unsqueeze(0).expand(size=(m, m)) - x @ x.T

    # Then we follow K(x, z) = exp(-1 * squareL2Diff / (2 * variance))
    return torch.exp(kernel / (-2 * variance))

def gaussianKernel(x1: torch.Tensor, x2: torch.Tensor, variance: float) -> float:
    ''' return the gaussian kernel dot product of 2 vectors, (n, 1) x (n, 1)

    Args:
        x1: (n, 1) first vector
        x2: (n, 1) second vector
        variance: variance of normal distribution
    Returns:
        K(x1, x2) where K = RBF
    '''
    assert(x2.size() == x1.size() and x2.dim() == 2 and x2.size()[1] == 0), f'expect 2 column vectors, received {x1.size()} and {x2.size()}'
    l2 = ((x2 - x1) ** 2).sum(dim=0)
    return torch.exp(-l2 / (2 * variance)).tolist()[0][0]

    

def trainPrediction(x: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor, b: float, kernel: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    ''' returns the prediction matrix U given x, y, alpha, b

    Args:
        x       (torch.Tensor): input matrix (m, n)
        y       (torch.Tensor): output vector (m, 1)
        alpha   (torch.Tensor): Lagrange parameter vecotr (m, 1)
        b       (float): y-intercept
        kernel  (Callable[[torch.Tensor], torch.Tensor]): Kernel function applied on all x[i], x[j]
    Returns:
        prediction matrix U (m, 1) where U[i] = SUM_{j = 1 .. n}{alpha[j] * y[j] * K(x[i], x[j])}
    '''
    m, _ = x.size()
    assert(y.size() == (m, 1)), f'y.size() = {y.size()} not (m, 1)'
    assert(alpha.size() == (m, 1)), f'alpha.size() = {alpha.size()} not (m, 1)'

    K = kernel(x)       # (m, m)
    Beta = alpha * y    # (m, 1)
    U = K @ Beta
    U += b
    return U

def errorVector(u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ''' returns the error vector E = U - Y s.t. E[i] = u(i) - y(i)

    Also handles size mismatch assertions

    Args:
        u: prediction vector, (m, 1)
        y: training vector, (m, 1)
    Returns:
        error vector u - y
    '''

    assert(u.size() == y.size()), f'size mismatch: u = {u.size()}, y = {y.size()}'
    assert(u.dim() == 1 or (u.dim() == 2 and u.size()[1] == 1)), f'expect collumn vector or array'

    return u - y

def prediction(
        xInput: torch.Tensor,
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: torch.Tensor, 
        b: float, 
        kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> float:
    ''' give confidence level of a prediction (negative = class -1, positive = class 1)

    Args:
        xInput: input array
        x: training set input
        y: training set output
        alpha: Lagrangian vector
        b: y-intercept (?)
        kernel: function on (m, n), (n, 1) (just use KBF in most cases)'''
    K = kernel(x, xInput)    #(m, 1)
    Beta = alpha * y        #(m, 1)
    U = (K.T @ Beta).flatten()
    U += b
    
    assert(U.size() == (1, 1)), 'input problems'
    
    return U.tolist()[0]


def updateLagrangians(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: torch.Tensor,
        E: torch.Tensor,
        C: float,
        i1: int,
        i2: int,
        kernel: Callable[[torch.Tensor, torch.Tensor], float],
) -> None:
    
    alpha1, alpha2 = float(alpha[i1, 0]), float(alpha[i2, 0])
    y1, y2 = float(y[i1, 0]), float(y[i2, 0])
    x1, x2 = x[i1], x[i2]
    E1, E2 = float(E[i1, 0]), float(E[i2, 0])

    # compute L,H bounds
    if y1 != y2:
        L = max(0, alpha2 - alpha1)
        H = min(C,     C + alpha2 - alpha1)
    else:
        L = max(0, alpha1 + alpha2 - C)
        H = min(C,     alpha1 + alpha2)
    if L == H:
        return


    eta = kernel(x1, x1) + kernel(x2, x2) - 2 * kernel(x1, x2)
    if eta <= 0:
        return

    alpha2New = alpha2 + y2 * (E1 - E2) / eta
    alpha2New = max(L, min(H, alpha2New))
    alpha1New = alpha1  + y1 * y2 * (alpha2 - alpha2New)

    alpha[i1, 0], alpha[i2, 0] = alpha1New, alpha2New
