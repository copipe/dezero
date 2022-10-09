import os
import subprocess

from dezero import Function, Variable


def _dot_var(v: Variable, verbose: bool = True) -> str:
    """Convert Variable information to dot language description.

    Args:
        v (Variable): Variable
        verbose (bool): If verbose is True, append shape and dtype information. Defaults to True.

    Returns:
        str: dot language description.
    """
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f: Function) -> str:
    """Convert Function information to dot language description.

    Args:
        f (Function): Function

    Returns:
        str: dot language description.
    """
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y is weakref
    return txt


def get_dot_graph(output: Variable, verbose=True) -> str:
    """Convert computational graph information to dot language description.

    Args:
        output (Variable): Final node of the computational graph.
        verbose (bool): If verbose is True, append shape and dtype information. Defaults to True.

    Returns:
        str: dot language description.
    """
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return "digraph g{\n" + txt + "}"


def plot_dot_graph(
    output: Variable, verbose: bool = True, to_file: str = "graph.png"
) -> None:
    """Export computational graph information to an image file.
    Args:
        output (Variable): Final node of the computational graph.
        verbose (bool): If verbose is True, append shape and dtype information. Defaults to True.
        to_file (str): Output image file name. Defaults to "graph.png".
    """
    dot_graph = get_dot_graph(output, verbose)

    # Save dot data to ~/.dezero/tmp_graph.dot.
    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    # Create image file with dot command.
    extension = os.path.splitext(to_file)[1][1:]
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)
