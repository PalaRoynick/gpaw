import os


class CLICommand:
    """Manipulate/show content of GPAW-restart file."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('gpw', metavar='gpw-file')

    @staticmethod
    def run(args):
        from gpaw.new.ase_interface import GPAW
        calc = GPAW(args.gpw)
        from gpaw.utilities.pointgroup_projections import SymmetryEigenvalues
        print(SymmetryEigenvalues.from_calc(calc, pani=True))
