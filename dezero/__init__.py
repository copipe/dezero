simple_core = True

if simple_core:
    from dezero.core_simple import (
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        using_config,
    )

    __all__ = [
        "Function",
        "Variable",
        "as_array",
        "as_variable",
        "no_grad",
        "using_config",
    ]
else:
    pass
