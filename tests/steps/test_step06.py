import numpy as np

from steps.step06 import Exp, Square, Variable


def test_grad():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    answer = 3.297442541400256
    assert abs(answer - x.grad) < 1e-6
