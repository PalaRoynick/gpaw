"""Just a logger to use while we develop."""
# FIXME/AP: delete this!

from ase.parallel import paropen
from traceback import format_stack
from inspect import currentframe
from datetime import datetime


def logg(message=None, traceback=False, header=True):
    """Writes messages to mylog.txt, and optionally can
    also put in a traceback. Parallel friendly."""
    now = datetime.now()
    with paropen('logg.txt', 'a') as f:
        if header is True:
            f.write('************* ' + now.isoformat() + ' ***************\n')
        if message is not None:
            f.write(str(message) + '\n')
        if traceback is True:
            stack_trace = format_stack(currentframe())
            for line in stack_trace:
                f.write(line.strip() + '\n')