
iimport pytest
import math
import torch

from models.supportVectorMachine import (
    predictionGaussianKernel,
    trainingGaussianKernel,
    gaussianKernel,
    trainPrediction,
    errorVector,
    prediction,
    updateLagrangians
)

def test_gaussianKernel_simple():
    x1 = torch.tensor([1.0, 2.0, 3.0])
    x2 = torch.tensor([1.0, 2.0, 3.0])
    assert predictionGaussianKernel.__module__  # ensure imported
    assert pytest.approx(1.0) == gaussianKernel(x1, x2, variance=5.0)

def test_trainingGaussianKernel_small():
    x = torch.tensor([[0.0], [1.0]])
    K = trainingGaussianKernel(x, variance=2.0)
    expected = torch.tensor([
        [1.0, math.exp(-1/4)],
        [math.exp(-1/4), 1.0]
    ])
    assert torch.allclose(K, expected, atol=1e-6)

def test_predictionGaussianKernel_vs_training():
    x_train = torch.tensor([[0.0], [2.0]])
    x_in = torch.tensor([[2.0]])
    var = 3.0
    full_K = trainingGaussianKernel(x_train, var)
    col_K = predictionGaussianKernel(x_train, x_in, var)
    assert torch.allclose(col_K.flatten(), full_K[:,1], atol=1e-6)

def test_trainPrediction_and_errorVector():
    X = torch.eye(3)
    y = torch.tensor([[1.0], [-1.0], [1.0]])
    alpha = torch.tensor([[1.0], [2.0], [3.0]])
    b = 0.5
    U = trainPrediction(X, y, alpha, b, kernel=lambda X: X @ X.T)
    expected_U = (alpha * y)
    expected_U += b
    assert torch.allclose(U, expected_U, atol=1e-6)
    E = errorVector(U, y)
    assert torch.allclose(E, U - y, atol=1e-6)

def test_prediction_consistency():
    X = torch.eye(2)
    y = torch.tensor([[1.0], [-1.0]])
    alpha = torch.tensor([[0.5], [0.5]])
    b = -0.1
    def lin_kernel(A, B): return A @ B.T
    U_train = trainPrediction(X, y, alpha, b, kernel=lambda X: X @ X.T)
    U_test = prediction(X, X, y, alpha, b, lin_kernel)
    assert torch.allclose(U_test, U_train, atol=1e-6)

def test_updateLagrangians_step():
    X = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[0.,1.,2.]])
    y = torch.tensor([[-1.],[1.],[1.],[-1.]])
    alpha = torch.tensor([[3.],[4.],[5.],[6.]], dtype=torch.float32)
    b = 2.0
    var = 4.0
    U = trainPrediction(X, y, alpha, b, kernel=lambda X: trainingGaussianKernel(X, var))
    E = errorVector(U, y)
    updateLagrangians(X, y, alpha, E, C=3.0, i1=3, i2=2,
                      kernel=lambda a,b: gaussianKernel(a, b, var))
    expected = torch.tensor([[3.0],[4.0],[0.0],[1.0]])
    assert torch.allclose(alpha, expected, atol=1e-6)
