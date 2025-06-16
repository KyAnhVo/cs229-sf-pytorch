import torch
import random
import numpy as np
from sklearn import svm as sksvm
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
        nonMoveCount = 0 
        NON_MOVE_COUNT_UNTIL_RANDOM = 20
        NON_MOVE_COUNT_UNTIL_TERMINATE = 100
        
        epochs = 0

        while nonMoveCount < NON_MOVE_COUNT_UNTIL_TERMINATE:
            print(epochs, '' if nonMoveCount < NON_MOVE_COUNT_UNTIL_RANDOM else 'randoming')
            epochs += 1

            i1, i2 = -1, -1
            
            # Choose indices logic
            if nonMoveCount < NON_MOVE_COUNT_UNTIL_RANDOM:
                i1, i2 = self._chooseIndices()
            else:
                while i1 == i2:
                    i1, i2 = random.randrange(self.m), random.randrange(self.m)

            if i1 == -1: # Converged
                break
            elif i2 == -1: # Data is very degenerate
                raise Exception('Degenerate data, will cause eta == 0')

            alpha1Prev = self.alpha[i1, 0].item()
            alpha2Prev = self.alpha[i2, 0].item()
            bPrev = self.b
            
            updateResult = self._updateLagrangians(i1, i2)
            if updateResult == LagrangeUpdateStatus.NO_MOVE_ETA_NULL:
                raise Exception('Kernel error or degenerate data')
            elif updateResult == LagrangeUpdateStatus.NO_MOVE_L_EQUALS_H:
                nonMoveCount += 1
                continue

            self._updateB(i1, i2, alpha1Prev, alpha2Prev)
            self._updateUAndE(i1, i2, alpha1Prev, alpha2Prev, bPrev)
        


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
        ''' choose 2 indices for optimizing Lagrangian.
    
        Let any number in [0, m) be denoted as t.
        Return cases for i1 and i2:
            (t, t):     2 indices to optimize
            (-1, -1):   Converged (can't find any index that violates KKT conditions
            (t, -1):    Data is degenerate: all x's are equivalent.
        '''
        def violatesKKT(index: int) -> bool:
            epsilon = 1e-3
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
            return i1, -1

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
        self.U += firstTerm + secondTerm - deltaB

        self.E[:] = self.U - self.y

##########################################################################################################
def test_smo_on_synthetic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)
    # 1) Make a tiny linearly separable 2D toy dataset:
    m = 100
    X_pos = np.random.randn(m//2, 2) + np.array([2, 2])
    X_neg = np.random.randn(m//2, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(m//2), -1*np.ones(m//2)]).reshape(-1, 1)

    # Convert to torch
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    # 2) Train your SVM (RBF)
    svm = SVM(x=X_t, y=y_t, variance=1.0, C=1.0)
    svm.train()

    # 3) Make predictions on the training set
    preds = np.array([np.sign(svm.prediction(torch.from_numpy(x.reshape(-1,1)).float().to(device)))
                      for x in X]).flatten()

    train_acc = (preds == y.flatten()).mean()
    print(f"SMO train acc on toy data: {train_acc:.3f}")

    # 4) Cross-check against scikit-learn’s SVC with the same kernel & params
    clf = sksvm.SVC(C=1.0, kernel='rbf', gamma=1.0/(2*1.0), tol=1e-3)
    clf.fit(X, y.flatten())
    sk_preds = clf.predict(X)
    sk_acc = (sk_preds == y.flatten()).mean()
    print(f"sklearn SVC train acc: {sk_acc:.3f}")

    

if __name__ == "__main__":
    test_smo_on_synthetic()

