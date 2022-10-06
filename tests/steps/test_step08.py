import numpy as np

from steps.step08 import Exp, Square, Variable


def test_backward():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    answer = 3.297442541400256
    assert abs(answer - x.grad) < 1e-6
