import numpy as np

from steps.step04 import Exp, Square, Variable, numerical_diff


def test_function1():
    f1 = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f1, x)
    answer = 4.0
    assert abs(answer - dy) < 1e-6


def test_function2():
    def f2(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = numerical_diff(f2, x)
    answer = 3.297442629333
    assert abs(answer - dy) < 1e-6
