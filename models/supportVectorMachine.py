import torch
from typing import Callable, Tuple


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

print('Test predictionGaussianKernel')
print(f'xTest = {xTest}\nxInputTest = {xInputTest.flatten()}')
print(f'gaussian kernel: {predictionGaussianKernel(xTest, xInputTest, variance= 4)}')
seperate()

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

print('Test trainingGaussianKernel')
print(f'xTest = {xTest}')
print(f'gaussian kernel: {trainingGaussianKernel(xTest, 4)}')
seperate()

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

print('Test gaussianKernel')
print('\n1. test on 1d array')
x1Test = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device)
x2Test = torch.tensor([1, 3, 5, 7, 9], dtype=torch.float32, device=device)
print(f'x1 = {x1Test.flatten()}\nx2 = {x2Test.flatten()}')
print(f'gaussian kernel: {gaussianKernel(x1Test, x2Test, 4)}')
print('\n2. test on 2d array')
x1Test = x1Test.unsqueeze(dim=1)
x2Test = x2Test.unsqueeze(dim=1)
print(f'x1 = {x1Test.flatten()}\nx2 = {x2Test.flatten()}')
print(f'gaussian kernel: {gaussianKernel(x1Test, x2Test, 4)}')
seperate()

#########################################################################################################################################

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
    return U.item()

print('Test prediction')
kernel = lambda x, y: predictionGaussianKernel(xTraining=x, xInput=y, variance=4)
print(f'input = {xInputTest.flatten()}\nx = {xTest}\ny = {yTest.flatten()}\nalpha = {alphaTest.flatten()}')
print(f'prediction is: {prediction(xInput=xInputTest, x=xTest, y=yTest, alpha=alphaTest, b=b, kernel=kernel)}')
seperate()

#########################################################################################################################################

##################################
### Functions dedicated to SMO ###
##################################

def trainPrediction(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    b: float,
    kernel: Callable[[torch.Tensor], torch.Tensor]
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

    K = kernel(x)       # (m, m)
    Beta = alpha * y    # (m, 1)
    U = K @ Beta
    U += b
    return U

print('Test trainPrediction')
print(f'input = {xInputTest}\nx = {xTest}\ny = {yTest.flatten()}\nalpha = {alphaTest.flatten()}\nb = {b}')
kernel = lambda x: trainingGaussianKernel(x, 4)
uTest = trainPrediction(x= xTest, y= yTest, alpha= alphaTest, b= b, kernel= kernel)
print(f'OUTPUT: U = {uTest.flatten()}')
seperate()

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

print('Test errorVector')
print(f'U = {uTest.flatten()}\nY = {yTest.flatten()}')
eTest = errorVector(uTest, yTest)
print(f'OUTPUT: E = {eTest.flatten()}')
seperate()

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

print('Test chooseIndices')

print('\nTest 1: base case alpha[i] == 0 for all i in [0, m)')
xTest = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype= torch.float32)
yTest = torch.tensor([[1], [-1], [-1], [1]], dtype= torch.float32)
b = 4.5
C = 6
alphaTest = torch.tensor([[0], [0], [0], [0]], dtype= torch.float32)
uTest = trainPrediction(xTest, yTest, alphaTest, b, kernel= lambda x: trainingGaussianKernel(x, 4))
eTest = errorVector(uTest, yTest)
i1, i2 = chooseIndices(xTest, yTest, alphaTest, uTest, eTest, C)
print(f'x: {xTest}\ny: {yTest}\nalpha: {alphaTest}\nb, C: {b, C}\nu: {uTest}\ne: {eTest}')
print(f'RESULT: i1, i2: {i1, i2}')

print('\n######')

print('\nTest 2: alpha[i] random, secondCache nonepty')
alphaTest = torch.tensor([[1], [-2], [3], [-10]], dtype= torch.float32)
uTest = trainPrediction(xTest, yTest, alphaTest, b, kernel= lambda x: trainingGaussianKernel(x, 4))
eTest = errorVector(uTest, yTest)
i1, i2 = chooseIndices(xTest, yTest, alphaTest, uTest, eTest, C)
print(f'x: {xTest}\ny: {yTest}\nalpha: {alphaTest}\nb, C: {b, C}\nu: {uTest}\ne: {eTest}')
print(f'RESULT: i1, i2: {i1, i2}')

print('\n######')

print('\nTest 3: alpha[i] random, secondCache empty')
alphaTest = torch.tensor([[-7], [-8], [9], [-10]])
uTest = trainPrediction(xTest, yTest, alphaTest, b, kernel= lambda x: trainingGaussianKernel(x, 4))
eTest = errorVector(uTest, yTest)
i1, i2 = chooseIndices(xTest, yTest, alphaTest, uTest, eTest, C)
print(f'x: {xTest}\ny: {yTest}\nalpha: {alphaTest}\nb, C: {b, C}\nu: {uTest}\ne: {eTest}')
print(f'RESULT: i1, i2: {i1, i2}')

print('\n######')

print('\nTest 4: no KKT‐violations (should return (-1,-1))')
alphaTest = torch.zeros(4,1)
# Manually pick U so that y*U >= 1 for all i, hence no KKT violations when alpha=0
uTest = torch.tensor([[ 1.0], [-1.0], [-1.0], [ 1.0]], dtype=torch.float32)
eTest = uTest - yTest
i1, i2 = chooseIndices(xTest, yTest, alphaTest, uTest, eTest, C)
print(f'x: {xTest}\ny: {yTest}\nalpha: {alphaTest}\nU: {uTest}\nE: {eTest}')
print(f'RESULT: i1, i2: {i1, i2}  # expect (-1, -1)')

print('\n######')

print('\nTest 5: third‐loop fallback (identical rows → no valid i2, expect (-1,-1))')
xTest = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype= torch.float32)
alphaTest = torch.zeros(4,1)  # first loop will pick i1 but no secondCache
uTest = torch.zeros(4,1)      # error values irrelevant
eTest = uTest - yTest
i1, i2 = chooseIndices(xTest, yTest, alphaTest, uTest, eTest, C)
print(f'x (all same): {xTest}\ny: {yTest}\nalpha: {alphaTest}\nU: {uTest}\nE: {eTest}')
print(f'RESULT: i1, i2: {i1, i2}  # expect (-1, -1)')

seperate()


#########################################################################################################################################

def updateLagrangians(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: torch.Tensor,
        E: torch.Tensor,
        C: float,
        i1: int,
        i2: int,
        kernel: Callable[[torch.Tensor, torch.Tensor], float]
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
        kernel: K(x1, x2)-> <K(x1), K(x2)>
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
        H = min(C,     C + alpha2 - alpha1)
    else:
        L = max(0, alpha1 + alpha2 - C)
        H = min(C,     alpha1 + alpha2)

    if L == H:
        # TODO: solve the L == H case.
        pass


    eta = kernel(x1, x1) + kernel(x2, x2) - 2 * kernel(x1, x2)

    # Should not happen if chosen kernel and i1, i2 are correct.
    if eta <= 0:
        return False

    alpha2New = alpha2 + y2 * (E1 - E2) / eta
    alpha2New = max(L, min(H, alpha2New))
    alpha1New = alpha1  + y1 * y2 * (alpha2 - alpha2New)
    alpha[i1, 0], alpha[i2, 0] = alpha1New, alpha2New
    return True

print('Test updateLagrangians')
C = 4 
i1, i2 = 2, 0
kernel = lambda x1, x2: gaussianKernel(x1, x2, 4)
print(f'Params: C = {C}, i1 = {i1}, i2 = {i2}')
print(f'x = {xTest}\ny = {yTest.flatten()}\nalpha = {alphaTest.flatten()}\nE = {eTest.flatten()}')
alpha1Prev, alpha2Prev = alphaTest[i1, 0].item(), alphaTest[i2, 0].item()
updateLagrangians(x= xTest, y= yTest, alpha= alphaTest, E= eTest, C= C, i1= i1, i2= i2, kernel= kernel)
print(f'\nnew Lagrangian vector:\nalpha = {alphaTest.flatten()}')
print(f'Old alphas: {alpha1Prev} {alpha2Prev}')
seperate()

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
    kernel: Callable[[torch.Tensor, torch.Tensor], float],
) -> float:
    
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

print('Test getB')
kernel = lambda x1, x2: gaussianKernel(x1, x2, 4)
print(f'Params: C = {C}, i1 = {i1}, i2 = {i2}, b = {b}')
print(f'x = {xTest}\ny = {yTest.flatten()}\nalpha = {alphaTest.flatten()}\nE = {eTest.flatten()}')
print(f'alpha1Prev = {alpha1Prev}, alpha2Prev = {alpha2Prev}')
b = updateB(alpha=alphaTest, x=xTest, y=yTest, b=b, E=eTest, C=C, i1=i1, i2=i2, alpha1Prev=alpha1Prev, alpha2Prev=alpha2Prev, kernel= lambda x1, x2: gaussianKernel(x1, x2, 4))
print(f'NEW B: b = {b}')
seperate()

#########################################################################################################################################
