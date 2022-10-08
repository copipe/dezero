import numpy as np

from dezero import Variable


def rozenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 10000

for i in range(iters):

    y = rozenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    x0.data -= lr * x0.grad  # type:ignore
    x1.data -= lr * x1.grad  # type:ignore

    print(x0.data, x1.data)
