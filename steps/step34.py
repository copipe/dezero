import matplotlib.pyplot as plt  # type:ignore
import numpy as np

import dezero.functions as F
from dezero import Variable

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data.flatten()]

for i in range(3):
    assert x.grad is not None
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.cleargrad()

    assert isinstance(gx, Variable)
    gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc="lower right")
plt.show()
