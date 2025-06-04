import torch
import math

# boilerplate stuffs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 300
m = 5000

# Kernel function

def gaussianKernel(x1: torch.Tensor, x2: torch.Tensor, variance: float) -> float:
    ''' returns the Gaussian (RBF) Kernel of the given column vectors
    
    Args:
        x1 (torch.Tensor):  first column vector
        x2 (torch.Tensor):  second column vector
        variance (float):   variance of Kernel
    Returns:
        dot product of the two kernelized vectors of x1 and x2.

    '''
    assert(x1.size()[1] == 1 and x2.size()[1] == 1), f'expect column vectors, received {x1.size()} and {x2.size()}'
    assert(x2.size() == x1.size()), f'expect vectors of equal length, received {x1.size()} and {x2.size()}'

    x1Flatten = x1.flatten()
    x2Flatten = x2.flatten()
    
    normSquared = (x1Flatten - x2Flatten).pow(2).sum()
    return torch.exp(-1 * normSquared / (2 * variance)).item()

