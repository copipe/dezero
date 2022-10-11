import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Variable
from dezero.models import Model


class TwoLayerNet(Model):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(in_size, hidden_size)
        self.l2 = L.Linear(hidden_size, out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


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

model = TwoLayerNet(in_size, hidden_size, out_size)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        assert p.grad is not None
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
