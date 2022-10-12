import numpy as np

import dezero.functions as F
from dezero import Variable, models, optimizers

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

model = models.MLP(in_size, [hidden_size, out_size])
# optimizer = optimizers.SGD(lr)
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)  # type:ignore
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)
