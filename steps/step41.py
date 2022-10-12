import numpy as np

import dezero.functions as F
from dezero import Variable

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
y.backward()

assert x.grad is not None
assert W.grad is not None
print(x.grad.shape)
print(W.grad.shape)
