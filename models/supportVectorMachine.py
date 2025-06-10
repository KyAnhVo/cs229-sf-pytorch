import torch
from typing import Callable

# boilerplate stuffs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seperate():
    print()
    print('-' * 100)
    print()

########################
### Kernel functions ###
########################

def predictionGaussianKernel(xTraining: torch.Tensor, xInput: torch.Tensor, variance: float) -> torch.Tensor:
    ''' returns the Gaussian (RBF) Kernel of the given column vectors, (m, n) x (n, 1)
    
    Args:
        xTraining (torch.Tensor):  training data (m, n)
        xInput (torch.Tensor):  second column vector (n, 1)
        variance (float):   variance of Kernel
    Returns:
        dot product of the two kernelized vectors of x1 and x2.

    '''
    m, n = xTraining.size()
    assert(xInput.size() == (n, 1)), f'input mismatch: {xTraining.size()}, {xInput.size()}'
    
    # Property: |x - y|^2 = |x|^2 + |y|^2 - 2<x, y>
    xInputL2 = (xInput.T ** 2).sum(dim=1, keepdim=True).expand(size=(m, 1))
    xTrainingL2 = (xTraining ** 2).sum(dim=1, keepdim=True)
    xDotX = xTraining @ xInput
    K = torch.exp(-(xInputL2 + xTrainingL2 - 2 * xDotX) / (2 * variance))
    return K

print('Test predictionGaussianKernel')
xTest = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype= torch.float32, device=device)
xInputTest = torch.tensor([[0, 2, 4, 6, 8]], dtype = torch.float32, device=device).T
print(f'xTest = {xTest.tolist()}\nxInputTest = {xInputTest.flatten().tolist()}')
print(f'gaussian kernel: {predictionGaussianKernel(xTest, xInputTest, variance= 4).tolist()}')
print('Exit test')
seperate()

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
    kernel = xLen.unsqueeze(1).expand(size=(m, m)) + xLen.unsqueeze(0).expand(size=(m, m)) - 2 * (x @ x.T)

    # Then we follow K(x, z) = exp(-1 * squareL2Diff / (2 * variance))
    return torch.exp(kernel / (-2 * variance))

print('Test trainingGaussianKernel')
xTest = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype= torch.float32, device=device)
print(f'xTest = {xTest.tolist()}')
print(f'gaussian kernel: {trainingGaussianKernel(xTest, 4).tolist()}')
print('Exit test')
seperate()

def gaussianKernel(x1: torch.Tensor, x2: torch.Tensor, variance: float) -> float:
    ''' return the gaussian kernel dot product of 2 vectors, (n, 1) x (n, 1)

    Args:
        x1: (n, 1) first vector
        x2: (n, 1) second vector
        variance: variance of normal distribution
    Returns:
        K(x1, x2) where K = RBF
    '''
    equivalent = x1.size() == x2.size()
    oneD = x1.dim() == 1
    twoD = x1.dim() == 2
    assert(equivalent), f'expect equivalent vectors, got x1 = {x1.size()}, x2 = {x2.size()}'
    assert(oneD or twoD), f'expect both to be 1D or 2D, received {x1.dim()}D'

    l2 = ((x1 - x2) ** 2).sum(dim=0)
    gauss = torch.exp(-1 * l2 / (2 * variance))
    return gauss.item()

print('Test gaussianKernel')
print('\n1. test on 1d array')
x1Test = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device)
x2Test = torch.tensor([1, 3, 5, 7, 9], dtype=torch.float32, device=device)
print(f'x1 = {x1Test}\nx2 = {x2Test}')
print(f'gaussian kernel: {gaussianKernel(x1Test, x2Test, 4)}')
print('\n2. test on 2d array')
x1Test = x1Test.unsqueeze(dim=1)
x2Test = x2Test.unsqueeze(dim=1)
print(f'x1 = {x1Test}\nx2 = {x2Test}')
print(f'gaussian kernel: {gaussianKernel(x1Test, x2Test, 4)}')
seperate()

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

##################################
### Functions dedicated to SMO ###
##################################

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
    ''' Given i1 and i2, update the Lagrangian

    Args:
        x: (m, n) x input
        y: (m, 1) y output where y[i] in {-1, 1}
        alpha: (m, 1) Lagrangian
        E: (m, 1) error vector E = U - Y
        C: diff 
        i1: first Lagrangian
        i2: second Lagrangian
        kernel: K(x1, x2)-> <K(x1), K(x2)>
    Returns:
        nothing, though alpha[i1] and alpha[i2] are changed.
    '''

    print('\nenter update lagrangian')
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
    print('exit update lagrangians\n')


