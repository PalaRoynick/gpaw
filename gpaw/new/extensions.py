from gpaw.extensions.d3 import D3 as _D3
import warnings


def D3(*args, **kwargs):
    warnings.warn('Please use "from gpaw.extensions.d3 import D3"',
                  DeprecationWarning)
    return _D3(*args, **kwargs)
