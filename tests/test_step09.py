import numpy as np
import pytest

from steps.step09 import Variable


@pytest.mark.parametrize(
    "data",
    # [np.array(0.5), None],
    [np.array(0.5)],
)
def test_valid_variable_type(data):
    x = Variable(data)
    assert isinstance(x, Variable)


def test_invalid_variable_type():
    with pytest.raises(TypeError) as e:
        _ = Variable(0.5)
    assert str(e.value) == "<class 'float'> is not supported."
