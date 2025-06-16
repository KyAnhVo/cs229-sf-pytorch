import torch
from typing import Tuple
from enum import Enum

class LagrangeUpdateStatus(Enum):
    UPDATED = 0             # updated successfully
    NO_MOVE_ETA_NULL = 1    # eta <= 0 error
    NO_MOVE_L_EQUALS_H = 2  # L == H degenerate box

class SVM:
    x: torch.Tensor
    y: torch.Tensor
    alpha: torch.Tensor
    U: torch.Tensor
    E: torch.Tensor
    K: torch.Tensor

    b: float
    C: float
    variance: float

    m: int
    n: int

    def __init__(self, x: torch.Tensor, y: torch.Tensor, variance, C):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert(x.dim() == 2 and y.dim() == 2), 'expect 2d matrices'
        assert(x.size()[0] == y.size()[0] and y.size()[1] == 1), 'expect (m, n) and (m, 1) matrices'

        self.x = x.to(dtype= torch.float32, device= device)
        self.y = y.to(dtype= torch.float32, device= device)
        self.m, self.n = x.size()

        self.alpha = torch.zeros(size= (self.m, 1), dtype= torch.float32, device= device)
        self.b = 0
        self.C = C
        self.variance = variance
        
        self._createTrainingGaussianKernel()
        self._initializeU()
        self.E = self.U - self.y

    def prediction(self, x: torch.Tensor) -> float:
        ''' give confidence level of a prediction (negative = class -1, positive = class 1)

        Args:
            x (torch.Tensor): input for prediction'''

        K = self._predictionGaussianKernel(x)
        Beta = self.alpha * self.y        #(m, 1)
        U = (K.T @ Beta).flatten()
        U += self.b
        return U.item()

    def train(self):
        # subject to change
        staticCount = 0 
        STATIC_COUNT_UNTIL_RANDOM = 20
        STATIC_COUNT_UNTIL_TERMINATE = 100

    def _gaussianKernel(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        ''' return the gaussian kernel dot product of 2 vectors, (n, 1) x (n, 1)

        Args:
            x1: (n, 1) first vector
            x2: (n, 1) second vector
        Returns:
            K(x1, x2) where K = RBF
        '''
        equivalent = x1.size() == x2.size()
        oneD = x1.dim() == 1
        twoD = x1.dim() == 2
        assert(equivalent), f'expect equivalent vectors, got x1 = {x1.size()}, x2 = {x2.size()}'
        assert(oneD or twoD), f'expect both to be 1D or 2D, received {x1.dim()}D'

        l2 = ((x1 - x2) ** 2).sum(dim=0)
        gauss = torch.exp(-1 * l2 / (2 * self.variance))
        return gauss.item()

    def _predictionGaussianKernel(self, x: torch.Tensor) -> torch.Tensor:
        ''' returns the matrix K s.t. K[i] = K(self.x[i], x)
    
        Args:
            x (torch.Tensor): new input vector
        Returns:
            matrix K s.t. K[i, 0] = K(self.x[i], x)

        '''
        assert(x.size() == (self.n, 1)), f'input mismatch: {self.x.size()}, {x.size()}'
    
        # Property: |x - y|^2 = |x|^2 + |y|^2 - 2<x, y>
        xInputL2 = (x.T ** 2).sum(dim=1, keepdim=True).expand(size=(self.m, 1))
        xTrainingL2 = (self.x ** 2).sum(dim=1, keepdim=True)
        xDotX = self.x @ x
        K = torch.exp(-(xInputL2 + xTrainingL2 - 2 * xDotX) / (2 * self.variance))
        return K

    def _createTrainingGaussianKernel(self) -> None:
        ''' Update the matrix K s.t. K[i][j] == K(x[i], x[j])'''

        # for each training set (row), we caclulate its l2 sqared length
        xLen = (self.x ** 2).sum(dim= 1) # (m,) 
    
        # We use the property |x - y|^2 = |x|^2 + |y|^2 - 2<x, y>
        # Applying to matrix, we havexLen
        # xLen(every row is the same) + xLen(every column is the same) - 2 * x @ X.T (m, m)
        # So each element [i][j] is |x(i) - x(j)|^2
        kernel = xLen.unsqueeze(1).expand(size=(self.m, self.m)) + xLen.unsqueeze(0).expand(size=(self.m, self.m)) - 2 * (self.x @ self.x.T)

        # Then we follow K(x, z) = exp(-1 * squareL2Diff / (2 * variance))
        self.K = torch.exp(kernel / (-2 * self.variance))

    def _initializeU(self):
        Beta = self.alpha * self.y    # (m, 1)
        self.U = self.K @ Beta
        self.U += self.b

    def _chooseIndices(self) -> Tuple[int, int]:
    
        def violatesKKT(index: int) -> bool:
            epsilon = 1e-8
            objective = self.y[index, 0].item() * self.U[index, 0].item()
            if 1 - epsilon <= objective and objective <= 1 + epsilon:
                return False
            currAlpha = self.alpha[index, 0].item()
            if currAlpha == 0:
                return objective < 1 - epsilon
            elif currAlpha == self.C:
                return objective > 1 + epsilon
            return True
    
        i1, i2 = -1, -1
        secondCache = []
    
        # First loop
        for i in range(self.m):
            if i1 == -1 and violatesKKT(i):
                i1 = i
                continue
            currAlpha = self.alpha[i, 0].item()
            if 0 < currAlpha and currAlpha < self.C:
                secondCache.append(i)
    
        if i1 == -1:
            return (-1, -1) # indicates that alphas are good already

        err1 = self.E[i1, 0].item()

        # Second loop
        secondCache = [i for i in secondCache if not torch.equal(self.x[i1], self.x[i])]
        if secondCache:
            i2 = max(secondCache, key= lambda i: abs(self.E[i, 0].item() - err1))
            return i1, i2
    
        # Third loop
        allValidIndexes = [i for i in range(self.m) if not self.x[i1].equal(self.x[i])]
        if not allValidIndexes:
            return -1, -1
        i2 = max(allValidIndexes, key= lambda i: abs(self.E[i, 0].item() - err1))
        return i1, i2

    def _updateLagrangians(self, i1, i2) -> LagrangeUpdateStatus:
        ''' Given i1 and i2, update the Lagrangian

        Args:
            i1: first Lagrangian
            i2: second Lagrangian
        Returns:
            old alpha[i1] and alpha[i2], though alpha[i1] and alpha[i2] are changed.
        '''

        alpha1, alpha2 = float(self.alpha[i1, 0]), float(self.alpha[i2, 0])
        y1, y2 = float(self.y[i1, 0]), float(self.y[i2, 0])
        E1, E2 = float(self.E[i1, 0]), float(self.E[i2, 0])

        # compute L,H bounds
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return LagrangeUpdateStatus.NO_MOVE_L_EQUALS_H

        eta = self.K[i1, i1].item() + self.K[i2, i2].item() - 2 * self.K[i1, i2].item()

        # Should not happen if chosen kernel and i1, i2 are correct.
        if eta <= 0:
            return LagrangeUpdateStatus.NO_MOVE_ETA_NULL

        alpha2New = alpha2 + y2 * (E1 - E2) / eta
        alpha2New = max(L, min(H, alpha2New))
        alpha1New = alpha1  + y1 * y2 * (alpha2 - alpha2New)
        self.alpha[i1, 0], self.alpha[i2, 0] = alpha1New, alpha2New

        return LagrangeUpdateStatus.UPDATED

    def _updateB(self, i1: int, i2: int, alpha1Prev: float, alpha2Prev: float) -> None:
        ''' Update b after changing the lagrangians.

        Args:
            i1 (int): first index of changed Lagrangian
            i2 (int): second index of changed Lagrangian
            alpha1Prev (float): previous value of first Lagrangian
            alpha2Prev (float): previous value of second Lagrangian
        '''

        a1, a2 = self.alpha[i1, 0].item(), self.alpha[i2, 0].item()
        y1, y2 = self.y[i1, 0].item(), self.y[i2, 0].item()
        E1, E2 = self.E[i1, 0].item(), self.E[i2, 0].item()
        k11, k22, k12 = self.K[i1, i1].item(), self.K[i2, i2].item(), self.K[i1, i2].item()

        b1 = E1 + y1 * (a1 - alpha1Prev) * k11 + y2 * (a2 - alpha2Prev) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1Prev) * k12 + y2 * (a2 - alpha2Prev) * k22 + self.b

        unbounded1 = 0 < a1 and a1 < self.C
        unbounded2 = 0 < a2 and a2 < self.C
    
        if unbounded1:
            self.b = b1
        elif unbounded2:
            self.b = b2
        else: # both bounded
            self.b = 0.5 * (b1 + b2)

    def _updateUAndE(self, i1: int, i2: int, alpha1Prev: float, alpha2Prev: float, bOld: float) -> None:
        y1, y2 = self.y[i1, 0].item(), self.y[i2, 0].item()
        deltaAlpha1 = self.alpha[i1, 0].item() - alpha1Prev
        deltaAlpha2 = self.alpha[i2, 0].item() - alpha2Prev
        deltaB = self.b - bOld

        firstTerm = y1 * deltaAlpha1 * self.K[:, i1].unsqueeze(dim= 1)
        secondTerm = y2 * deltaAlpha2 * self.K[:, i2].unsqueeze(dim= 1)
        self.U += firstTerm + secondTerm + deltaB

        self.E[:] = self.U - self.y



