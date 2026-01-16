from gpaw import GPAW_NO_C_EXTENSION
from types import ModuleType

# Import everything from the C-extension module, or the purepython equivalent.
# Other modules should use `from gpaw.cgpaw import xxx` to access stuff
# from the extension module.

if GPAW_NO_C_EXTENSION:
    from gpaw.purepython import *  # noqa: F401, F403
else:
    from _gpaw import *  # noqa: F401, F403

    # Ensure we have symbols that were added to C-extension without bumping
    # its version. Bad practice, obviously. This is a hotfix
    try:
        have_magma_symbols = "have_magma" in globals()
    except NameError:
        have_magma = False


def _get_extension_module() -> ModuleType | None:
    """Return direct access to the _gpaw extension module, if available.
    Internal use only.
    """
    if GPAW_NO_C_EXTENSION:
        return None
    else:
        import _gpaw
        return _gpaw
