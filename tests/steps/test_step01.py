import numpy as np

from steps.step01 import Variable


def test_variable():
    data = np.array(1.0)
    x = Variable(data)
    assert x.data == np.array(1.0)
