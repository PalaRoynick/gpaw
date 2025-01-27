import numpy as np


def object_to_string(obj):
    if isinstance(obj, np.ndarray):
        return f'ndarray(shape={obj.shape}, dtype={obj.dtype})'
    return str(obj)


def recursive_listing(reader, indent=0, _indent_size=4):
    import ase.io.ulm as ulm
    nextindent = indent + _indent_size

    for key in reader.keys():
        obj = getattr(reader, key)
        spaces = ' ' * indent
        if isinstance(obj, ulm.Reader):
            print(f'{spaces}{key}')
            recursive_listing(obj, indent=nextindent)
        else:
            string = object_to_string(obj)
            print(f'{spaces}{key}: {string}')


def print_gpw_info(filename, reader):
    print(filename)
    recursive_listing(reader, indent=4)


class CLICommand:
    """Display contents of GPAW-restart file."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('gpw', metavar='gpw-file')

    @staticmethod
    def run(args):
        import ase.io.ulm as ulm
        with ulm.open(args.gpw) as reader:
            print_gpw_info(args.gpw, reader)
