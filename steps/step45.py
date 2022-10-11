import numpy as np

import dezero.functions as F
from dezero import Variable
from dezero.models import MLP

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x = Variable(x)  # type:ignore
y = Variable(y)


lr = 0.2
iters = 10000
in_size = 1
hidden_size = 10
out_size = 1

model = MLP(in_size, [hidden_size, out_size])

for i in range(iters):
    y_pred = model(x)  # type:ignore
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        assert p.grad is not None
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
