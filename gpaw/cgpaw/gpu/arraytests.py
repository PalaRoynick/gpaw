try:
    from _gpaw.gpu.arraytests import *  # noqa: F401, F403
    have_magma = True
except (ImportError, ModuleNotFoundError):
    pass
