import numpy as np
import pytest

from steps.step12 import Variable, add


@pytest.mark.parametrize(
    "x1, x2, answer",
    [
        (Variable(np.array(2.0)), Variable(np.array(3.0)), 5.0),
        (Variable(np.array(-2.0)), Variable(np.array(2.0)), 0.0),
    ],
)
def test_add(x1, x2, answer):
    y = add(x1, x2)
    assert y.data == answer
