import numpy as np

from dezero import Variable

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.reshape((1, 6))
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.T
y.backward()
print(y)
print(x.grad)
