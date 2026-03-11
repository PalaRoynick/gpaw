import functools
import inspect

import numpy as np

from gpaw import GPAW_NO_C_EXTENSION

if GPAW_NO_C_EXTENSION:
    from gpaw.purepython import *  # noqa: F401, F403
else:
    import _gpaw as _cgpaw_ext

    from _gpaw import *  # noqa: F401, F403
    _GPAW_IS_CPP = getattr(_cgpaw_ext, 'gpaw_cpp', False)

    _PYBIND_SINGLE_PRECISION_SAFE = set()

    if _GPAW_IS_CPP:
        _pw = _cgpaw_ext.pw  # type: ignore
        add_to_density = _pw.add_to_density
        pawexxvv = _pw.pawexxvv
        plane_wave_grid = _pw.plane_wave_grid
        pw_insert = _pw.pw_insert
        pw_precond = _pw.pw_precond
        pwlfc_expand = _pw.pwlfc_expand
        pwlfc_expand_old = _pw.pwlfc_expand_old
        try:
            mmm = _pw.mmm
        except ImportError:
            pass
        except AttributeError:
            pass
        try:
            r2k = _pw.r2k
        except ImportError:
            pass
        except AttributeError:
            pass

        _PYBIND_SINGLE_PRECISION_SAFE = {
            'mmm',
            'r2k',
            'add_to_density',
            'pawexxvv',
            'plane_wave_grid',
            'pw_insert',
            'pw_precond',
            'pwlfc_expand',
            'pwlfc_expand_old',
        }

    def _contains_single_precision(obj) -> bool:
        if isinstance(obj, np.ndarray):
            return obj.dtype in (np.float32, np.complex64)
        if isinstance(obj, np.generic):
            return obj.dtype in (np.float32, np.complex64)
        if isinstance(obj, (list, tuple)):
            return any(_contains_single_precision(value) for value in obj)
        if isinstance(obj, dict):
            return any(_contains_single_precision(value)
                       for value in obj.values())
        return False

    def _wrap_legacy_c_function(name, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if (_contains_single_precision(args) or
                    _contains_single_precision(kwargs)):
                raise RuntimeError(
                    f"Legacy C-extension function '{name}' does not support "
                    "float32/complex64 inputs. This call reached the old "
                    "_gpaw path; migrate this function to _gpaw.pw next "
                    "or upcast inputs to double precision.")
            return func(*args, **kwargs)

        return wrapper

    for _name, _obj in list(globals().items()):
        if (_name.startswith('_') or
                _name in _PYBIND_SINGLE_PRECISION_SAFE or
                not inspect.isbuiltin(_obj)):
            continue
        globals()[_name] = _wrap_legacy_c_function(_name, _obj)


def get_extension_module_path() -> str:
    """Return path to the _gpaw extension module. If the extension module is
    not available, returns path to a 'purepython' module that is used in
    place of the actual extension.
    """
    from pathlib import Path
    if GPAW_NO_C_EXTENSION:
        import gpaw.purepython as mod  # type:ignore
    else:
        import _gpaw as mod  # type:ignore
    return str(Path(mod.__file__).resolve())
