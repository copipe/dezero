import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Variable

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x = Variable(x)  # type:ignore
y = Variable(y)

I, H, O_ = 1, 10, 1
l1 = L.Linear(I, H)
l2 = L.Linear(H, O_)


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l_ in [l1, l2]:
        for p in l_.params():
            assert p.grad is not None
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
