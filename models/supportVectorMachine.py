import torch
from typing import Tuple
from enum import Enum

class UpdateVal(Enum):
    SUCCESS = 0
    ETA_ERROR = 1
    L_EQUALS_H = 2


# boilerplate stuffs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seperate():
    print()
    print('-' * 100)
    print()

########################
### Test inputs ########
########################

xInputTest = torch.tensor([[0, 2, 4, 6, 8]], dtype = torch.float32, device=device).T
xTest = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype= torch.float32, device=device)
yTest = torch.tensor([[-1], [-1], [1]], dtype=torch.float32, device=device)
alphaTest = torch.tensor([[2], [4], [1]], dtype=torch.float32, device=device)
b = 4.5

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

#########################################################################################################################################

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
    # xLen(every row is the same) + xLen(every column is the same) - 2 * x @ X.T (m, m)
    # So each element [i][j] is |x(i) - x(j)|^2
    kernel = xLen.unsqueeze(1).expand(size=(m, m)) + xLen.unsqueeze(0).expand(size=(m, m)) - 2 * (x @ x.T)

    # Then we follow K(x, z) = exp(-1 * squareL2Diff / (2 * variance))
    return torch.exp(kernel / (-2 * variance))

#########################################################################################################################################

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

#########################################################################################################################################

def prediction(
        xInput: torch.Tensor,
        x: torch.Tensor, 
        y: torch.Tensor, 
        alpha: torch.Tensor, 
        b: float, 
        variance: float,        
) -> float:
    ''' give confidence level of a prediction (negative = class -1, positive = class 1)

    Args:
        xInput: input array
        x: training set input
        y: training set output
        alpha: Lagrangian vector
        b: y-intercept (?)
        kernel: function on (m, n), (n, 1) (just use KBF in most cases)'''

    K = predictionGaussianKernel(x, xInput, variance)    #(m, 1)
    Beta = alpha * y        #(m, 1)
    U = (K.T @ Beta).flatten()
    U += b
    return U.item()

#########################################################################################################################################

##################################
### Functions dedicated to SMO ###
##################################

def trainPrediction(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    b: float,
    variance: float,
) -> torch.Tensor:
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

    K = trainingGaussianKernel(x, variance)       # (m, m)
    Beta = alpha * y    # (m, 1)
    U = K @ Beta
    U += b
    return U

#########################################################################################################################################

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

#########################################################################################################################################

def chooseIndices(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    U: torch.Tensor,
    E: torch.Tensor,
    C: float
) -> Tuple[int, int]:
    
    def violatesKKT(index: int) -> bool:
        epsilon = 1e-8
        objective = y[index, 0].item() * U[index, 0].item()
        if 1 - epsilon <= objective and objective <= 1 + epsilon:
            return False
        currAlpha = alpha[index, 0].item()
        if currAlpha == 0:
            return objective < 1 - epsilon
        elif currAlpha == C:
            return objective > 1 + epsilon
        return True
    
    m, _ = alpha.size()
    i1, i2 = -1, -1
    secondCache = []
    
    # First loop
    for i in range(m):
        if i1 == -1 and violatesKKT(i):
            i1 = i
            continue
        currAlpha = alpha[i, 0].item()
        if 0 < currAlpha and currAlpha < C:
            secondCache.append(i)
    
    if i1 == -1:
        return (-1, -1) # indicates that alphas are good already

    err1 = E[i1, 0].item()

    # Second loop
    secondCache = [i for i in secondCache if not torch.equal(x[i1], x[i])]
    if secondCache:
        i2 = max(secondCache, key= lambda i: abs(E[i, 0].item() - err1))
        return i1, i2
    
    # Third loop
    allValidIndexes = [i for i in range(m) if not x[i1].equal(x[i])]
    if not allValidIndexes:
        return -1, -1
    i2 = max(allValidIndexes, key= lambda i: abs(E[i, 0].item() - err1))
    return i1, i2

#########################################################################################################################################

def updateLagrangians(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: torch.Tensor,
        E: torch.Tensor,
        C: float,
        i1: int,
        i2: int,
        variance: float
) -> bool:
    ''' Given i1 and i2, update the Lagrangian

    Args:
        x: (m, n) x input
        y: (m, 1) y output where y[i] in {-1, 1}
        alpha: (m, 1) Lagrangian
        E: (m, 1) error vector E = U - Y
        C: diff 
        i1: first Lagrangian
        i2: second Lagrangian
        variance: variance of Gauss kernel
    Returns:
        old alpha[i1] and alpha[i2], though alpha[i1] and alpha[i2] are changed.
    '''

    alpha1, alpha2 = float(alpha[i1, 0]), float(alpha[i2, 0])
    y1, y2 = float(y[i1, 0]), float(y[i2, 0])
    x1, x2 = x[i1], x[i2]
    E1, E2 = float(E[i1, 0]), float(E[i2, 0])

    # compute L,H bounds
    if y1 != y2:
        L = max(0, alpha2 - alpha1)
        H = min(C, C + alpha2 - alpha1)
    else:
        L = max(0, alpha1 + alpha2 - C)
        H = min(C, alpha1 + alpha2)

    if L == H:
        return False

    kernel = lambda x1, x2: gaussianKernel(x1, x2, variance)
    eta = kernel(x1, x1) + kernel(x2, x2) - 2 * kernel(x1, x2)

    # Should not happen if chosen kernel and i1, i2 are correct.
    if eta <= 0:
        return False

    alpha2New = alpha2 + y2 * (E1 - E2) / eta
    alpha2New = max(L, min(H, alpha2New))
    alpha1New = alpha1  + y1 * y2 * (alpha2 - alpha2New)
    alpha[i1, 0], alpha[i2, 0] = alpha1New, alpha2New
    return True

#########################################################################################################################################

def updateB(
    alpha: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    b: float,
    E: torch.Tensor,
    C: float,
    i1: int,
    i2: int,
    alpha1Prev: float,
    alpha2Prev: float,
    variance: float,
) -> float:
    kernel = lambda x1, x2: gaussianKernel(x1, x2, variance)

    a1, a2 = alpha[i1, 0].item(), alpha[i2, 0].item()
    x1, x2 = x[i1], x[i2]
    y1, y2 = y[i1, 0].item(), y[i2, 0].item()
    E1, E2 = E[i1, 0].item(), E[i2, 0].item()
    k11, k22, k12 = kernel(x1, x1), kernel(x2, x2), kernel(x1, x2)

    b1 = E1 + y1 * (a1 - alpha1Prev) * k11 + y2 * (a2 - alpha2Prev) * k12 + b
    b2 = E2 + y1 * (a1 - alpha1Prev) * k12 + y2 * (a2 - alpha2Prev) * k22 + b

    unbounded1 = 0 < a1 and a1 < C
    unbounded2 = 0 < a2 and a2 < C
    
    if unbounded1:
        return b1
    elif unbounded2:
        return b2
    else: # both bounded
        return 0.5 * (b1 + b2)

#########################################################################################################################################

def updateUE(
        U: torch.Tensor,
        E: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: float,
        y2: float,
        deltaAlpha1: float,
        deltaAlpha2: float,
        deltaB: float,
        variance: float,
) -> None:
    alpha1Term = y1 * deltaAlpha1 * predictionGaussianKernel(x, x1, variance)
    alpha2Term = y2 * deltaAlpha2 * predictionGaussianKernel(x, x2, variance)
    U += alpha1Term + alpha2Term - deltaB
    E[:] = errorVector(u= U, y= y)

#########################################################################################################################################

def SVM(x: torch.Tensor, y: torch.Tensor, C: float, variance: float) -> Tuple[torch.Tensor, float]:
    ''' Returns alpha vector and b for an SVM implementation

    Args:
        x (torch.Tensor): (m, n) matrix of inputs.
        y (torch.Tensor): (m, 1) matrix of output where y[i] in {-1, 1} for all i.
        C (float): regularization module
        variance (float): variance of the Gaussian kernel
    Returns:
        (alpha, b): used for using prediction() to call functions
    '''

    assert(x.dim() == 2), 'expect matrix x, i.e. 2D tensor.'
    m, n = x.size()
    assert(y.dim() == 2 and y.size() == (m, 1)), 'expect column vecotr y, i.e. y is a (m, 1) matrix'

    alpha = torch.zeros(size= (m, 1), dtype= torch.float32)
    b = 0
    U = trainPrediction(x, y, alpha, b, variance)
    E = errorVector(u= U, y= y)

    # the 2 values are arbitrary, should be tuned
    currentNonupdate = 0
    MAX_NONUPDATE_TILL_RANDOM_SELECTION = 7
    MAX_NONUPDATE_TILL_CONVERGENCE = 10

    return alpha, b
