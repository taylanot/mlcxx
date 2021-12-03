import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict
from sacred import Experiment
# The Dict.empty() constructs a typed dictionary.
# The key and value typed must be explicitly declared.
ex = Experiment('run')
@ex.config
def my_config():
    d = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )

    # The typed-dict can be used from the interpreter.
    d['x'] = np.asarray([1, 0.5, 2], dtype='f8')
    d['y'] = np.asarray([1], dtype='f8')


# Here's a function that expects a typed-dict as the argument
@njit
def move(d):
    # inplace operations on the arrays
    for k,v in d.items():
        v += 1
    return d
# Call move(d) to inplace update the arrays in the typed-dict.

@ex.automain
def main(d):
    print(move(d))

