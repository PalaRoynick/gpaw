from ase.io import read

from gpaw.io.log_file_reader import parse_str
from gpaw.new.logger import RED, RESET


def test_gpaw_log(gpw_files):
    path = gpw_files['h2_pw'].with_suffix('.txt')
    atoms = read(path)
    assert len(atoms) == 2


def test_colors():
    assert parse_str(RED + '-1.234' + RESET) == -1.234
