"""GPAW command-line tool."""
import os
import subprocess
import sys


commands = [
    ('run', 'gpaw.cli.run'),
    ('info', 'gpaw.cli.info'),
    ('test', 'gpaw.cli.test'),
    ('dos', 'gpaw.cli.dos'),
    ('gpw', 'gpaw.cli.gpw'),
    ('completion', 'gpaw.cli.completion'),
    ('atom', 'gpaw.atom.aeatom'),
    ('diag', 'gpaw.cli.fulldiag'),
    # ('quick', 'gpaw.cli.quick'),
    ('python', 'gpaw.cli.python'),
    ('sbatch', 'gpaw.cli.sbatch'),
    ('dataset', 'gpaw.atom.generator2'),
    ('plot-dataset', 'gpaw.atom.plot_dataset'),
    ('basis', 'gpaw.atom.basisfromfile'),
    ('plot-basis', 'gpaw.basis_data'),
    ('symmetry', 'gpaw.symmetry'),
    ('install-data', 'gpaw.cli.install_data')]


def hook(parser, args):
    parser.color = True
    parser.suggest_on_error = True
    parser.add_argument('-P', '--parallel', type=int, metavar='N',
                        help='Run on N CPUs.')
    parser.add_argument('--xxx', action='store_true')

    parser.color = True
    args = parser.parse_args(args)

    if args.command == 'python':
        args.traceback = True

    if hasattr(args, 'dry_run'):
        N = int(args.dry_run)
        if N:
            import gpaw
            gpaw.dry_run = N
            import gpaw.mpi as mpi
            mpi.world = mpi.SerialCommunicator()
            mpi.world.size = N

    if not args.xxx:
        from gpaw.mpi import have_mpi, world
        if not have_mpi:
            raise SystemExit('MPI not available')

        if 1:#py:
            # Start again in parallel:
            pyargs = []
            if sys.version_info >= (3, 11):
                # Don't prepend a potentially unsafe path to sys.path
                pyargs.append('-P')
            if args.warnings:
                pyargs += ['-W', args.warnings]
            print(args)
            arguments = ['mpiexec',
                         *os.environ.get('GPAW_MPI_OPTIONS', '').split(),
                         '-np',
                         str(args.parallel or 1),
                         sys.executable,
                         *pyargs,
                         '-m',
                         'gpaw', '--xxx',
                         *sys.argv[1:]]
            print(arguments)

            # Use a clean set of environment variables without any MPI stuff:
            p = subprocess.run(arguments, check=False, env=os.environ)
            sys.exit(p.returncode)
        else:
            lkjhasdfljkh

    return args


def main(args=None):
    from gpaw import all_lazy_imports, broadcast_imports, __getattr__
    with broadcast_imports:
        for attr in all_lazy_imports:
            __getattr__(attr)

        from ase.cli.main import main as ase_main
        from gpaw import __version__

    ase_main('gpaw', 'GPAW command-line tool', __version__,
             commands, hook, args)
