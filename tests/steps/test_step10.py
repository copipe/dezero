import numpy as np
import pytest

from steps.step10 import Add, Variable


class TestAdd:
    @pytest.mark.parametrize(
        "xs, answer",
        [
            ([Variable(np.array(2.0)), Variable(np.array(3.0))], 5.0),
            ([Variable(np.array(-2.0)), Variable(np.array(2.0))], 0.0),
        ],
    )
    def test_foward(self, xs, answer):
        f = Add()
        ys = f(xs)
        y = ys[0]
        assert y.data == answer
