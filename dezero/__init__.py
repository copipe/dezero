simple_core = False

if simple_core:
    from dezero.core_simple import (
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        using_config,
    )
else:
    from dezero.core import (
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        using_config,
    )

    pass

__all__ = [
    "Function",
    "Variable",
    "as_array",
    "as_variable",
    "no_grad",
    "using_config",
]
