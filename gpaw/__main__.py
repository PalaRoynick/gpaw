import gpaw
from gpaw.cli.main import main

# Set MPI backend to 'cgpaw' by default if executing
# `gpaw` or `python -m gpaw`
if gpaw.GPAW_MPI_BACKEND is None:
    import gpaw.cgpaw as cgpaw
    if cgpaw.have_mpi:
        gpaw.GPAW_MPI_BACKEND = "cgpaw"
    else:
        gpaw.GPAW_MPI_BACKEND = "serial"

    # Logic for keeping deprecated -P/--parallel working
    # TODO: remove this when the -P option is removed
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--parallel', nargs='?', const=1, type=int)
    args, _ = parser.parse_known_args([sys.argv[1]])
    if args.parallel is not None:
        gpaw.GPAW_MPI_BACKEND = "serial"
    # End of logic

if __name__ == '__main__':
    main()
