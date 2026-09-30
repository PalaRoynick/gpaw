import gpaw
from gpaw.cli.main import main


def figure_out_mpi_backend():
    # Logic for keeping deprecated -P/--parallel working
    # TODO: remove this when the -P option is removed
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--parallel', nargs='?', const=1, type=int)
    args, _ = parser.parse_known_args([sys.argv[1]])
    if args.parallel is not None:
        return 'serial'
    # End of logic

    try:
        import mpi4py  # noqa

        return 'mpi4py'
    except ImportError:
        pass

    import gpaw.cgpaw as cgpaw

    if cgpaw.have_mpi:
        return "cgpaw"
    else:
        return "serial"


# Set MPI backend to 'cgpaw' by default if executing
# `gpaw` or `python -m gpaw`
if gpaw.GPAW_MPI_BACKEND is None:
    gpaw.GPAW_MPI_BACKEND = figure_out_mpi_backend()

if __name__ == '__main__':
    main()
