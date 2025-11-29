from ase.collections import g2
import pytest
from ase.io import write
from ase.build import molecule
from pathlib import Path
from gpaw.utilities.pointgroup import PointGroup, Projectable, SPGOperations
from gpaw.new.ase_interface import GPAW
from dataclasses import dataclass
from numpy import pi, sin, cos

import os
import contextlib

from itertools import zip_longest


@contextlib.contextmanager
def workdir(path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def analyze_symmetry(calc, layergroup, expected):
    spg_ops = SPGOperations.from_atoms(calc.atoms, layergroup=layergroup)
    assert spg_ops.pointgroup.upper() == expected.upper()
    pg = PointGroup(spg_ops, [0, 0, 1] if layergroup else None)
    results = []
    failure = False
    for band in range(6):
        signature = pg.signature(Projectable.from_calc(calc, band))
        found = None
        for irrep, s in zip(
            pg.character_table.irreps,
            pg.detect_irrep(signature),
        ):
            if s > 0.01:
                print(band, irrep, f"{s.real:.2f}")
                results.append(irrep)
                if found is not None:
                    failure = True
                found = irrep
    if failure:
        raise ValueError("Band spans multiple irreps.")
    return results


turbomole_input = """

a coord
desy
*
no
b
all DZP
*
eht
y
0
y
dft
func
pbe
on
*
*
"""


@dataclass
class State:
    irrep: str
    eigenvalue: float
    occupation: float
    degeneracy: int = 1

    def __format__(self, fmt):
        return f"{self.irrep:5s} {self.eigenvalue:8.2f} {self.occupation:5.2f}"


def grouped_tokens(tokens):
    for t in tokens:
        # Case 1: token is a float -> (1, value)
        if "." in t:
            yield (1, float(t))

        # Case 2: token is an int -> expect "*", then float
        else:
            count = int(t)

            star = next(tokens)  # must be "*"
            if star != "*":
                raise ValueError(f"Expected '*', got {star!r}")

            value_token = next(tokens)  # must be a float string
            if "." not in value_token:
                raise ValueError(f"Expected float token, got {value_token!r}")

            yield (count, float(value_token))


def parse_eigenvalues(text):
    print(text)
    lines = iter(text.split("\n"))

    states = []
    try:
        while True:
            irreps = iter(next(lines).split())
            Hartree = iter(next(lines).split())
            eV = iter(next(lines).split())
            occupations = iter(next(lines).split())
            assert next(irreps) == "irrep"
            assert next(Hartree) == "eigenvalues"
            assert next(Hartree) == "H"
            assert next(eV) == "eV"
            try:
                assert next(occupations) == "occupation"
            except StopIteration:
                # If there are no occupations, there are no text in this line
                pass
            for irr, eig, (deg, occ) in zip_longest(
                irreps, eV, grouped_tokens(occupations), fillvalue=(1, 0.0)
            ):
                states.append(State(irr, float(eig), float(occ), deg))
    except StopIteration:
        pass
    return sorted(states, key=lambda state: state.eigenvalue)


systems = {
    "PH3": "cs",
    "P2": "d6h",
    "CH3CHO": "cs",
    "H2COH": "c1",
    "CS": "c6v",
    "OCHCHO": "c2h",
    "C3H9C": "cs",
    "CH3COF": "cs",
    "CH3CH2OCH3": "cs",
    "HCOOH": "cs",
    "HCCl3": "c3v",
    "HOCl": "cs",
    "H2": "d6h",
    "SH2": "c2v",
    "C2H2": "d6h",
    "C4H4NH": "c2v",
    "CH3SCH3": "c2v",
    "SiH2_s3B1d": "c2v",
    "CH3SH": "cs",
    "CH3CO": "cs",
    "CO": "c6v",
    "ClF3": "c2v",
    "SiH4": "td",
    "C2H6CHOH": "c1",
    "CH2NHCH2": "cs",
    "isobutene": "c2v",
    "HCO": "cs",
    "bicyclobutane": "c2v",
    "LiF": "c6v",
    "Si": "oh",
    "C2H6": "c2h",
    "CN": "c6v",
    "ClNO": "cs",
    "S": "oh",
    "SiF4": "td",
    "H3CNH2": "cs",
    "methylenecyclopropane": "c2v",
    "CH3CH2OH": "cs",
    "F": "oh",
    "NaCl": "c6v",
    "CH3Cl": "cs",
    "CH3SiH3": "c3v",
    "AlF3": "d3h",
    "C2H3": "cs",
    "ClF": "c6v",
    "PF3": "c3v",
    "PH2": "c2v",
    "CH3CN": "c3v",
    "cyclobutene": "c2v",
    "CH3ONO": "cs",
    "SiH3": "cs",
    "C3H6_D3h": "d3h",
    "CO2": "d6h",
    "NO": "c6v",
    "trans-butane": "c2h",
    "H2CCHCl": "cs",
    "LiH": "c6v",
    "NH2": "c2v",
    "CH": "c6v",
    "CH2OCH2": "c2v",
    "C6H6": "d6h",
    "CH3CONH2": "c1",
    "cyclobutane": "d2d",
    "H2CCHCN": "cs",
    "butadiene": "c2h",
    "C": "oh",
    "H2CO": "c2v",
    "CH3COOH": "cs",
    "HCF3": "c3v",
    "CH3S": "cs",
    "CS2": "d6h",
    "SiH2_s1A1d": "c2v",
    "C4H4S": "c2v",
    "N2H4": "c2",
    "OH": "c6v",
    "CH3OCH3": "c2v",
    "C5H5N": "c2v",
    "H2O": "c2v",
    "HCl": "c6v",
    "CH2_s1A1d": "c2v",
    "CH3CH2SH": "cs",
    "CH3NO2": "cs",
    "Cl": "oh",
    "Be": "oh",
    "BCl3": "d3h",
    "C4H4O": "c2v",
    "Al": "oh",
    "CH3O": "cs",
    "CH3OH": "cs",
    "C3H7Cl": "cs",
    "isobutane": "cs",
    "Na": "oh",
    "CCl4": "td",
    "CH3CH2O": "cs",
    "H2CCHF": "cs",
    "C3H7": "cs",
    "CH3": "c2v",
    "O3": "c2v",
    "P": "oh",
    "C2H4": "d2h",
    "NCCN": "d6h",
    "S2": "d6h",
    "AlCl3": "d3h",
    "SiCl4": "td",
    "SiO": "c6v",
    "C3H4_D2d": "d2d",
    "H": "oh",
    "COF2": "c2v",
    "2-butyne": "c2v",
    "C2H5": "cs",
    "BF3": "d3h",
    "N2O": "c6v",
    "F2O": "c2v",
    "SO2": "c2v",
    "H2CCl2": "c2v",
    "CF3CN": "cs",
    "HCN": "c6v",
    "C2H6NH": "cs",
    "OCS": "c6v",
    "B": "oh",
    "ClO": "c6v",
    "C3H8": "c2v",
    "HF": "c6v",
    "O2": "d6h",
    "SO": "c6v",
    "NH": "c6v",
    "C2F4": "d2h",
    "NF3": "c3v",
    "CH2_s3B1d": "c2v",
    "CH3CH2Cl": "cs",
    "CH3COCl": "cs",
    "NH3": "cs",
    "C3H9N": "cs",
    "CF4": "td",
    "C3H6_Cs": "cs",
    "Si2H6": "d3d",
    "HCOOCH3": "cs",
    "O": "oh",
    "CCH": "c6v",
    "N": "oh",
    "Si2": "d6h",
    "C2H6SO": "c1",
    "C5H8": "d2d",
    "H2CF2": "c2v",
    "Li2": "d6h",
    "CH2SCH2": "c2v",
    "C2Cl4": "d2h",
    "C3H4_C3v": "cs",
    "CH3COCH3": "c2v",
    "F2": "d6h",
    "CH4": "td",
    "SH": "c6v",
    "H2CCO": "c2v",
    "CH3CH2NH2": "cs",
    "Li": "oh",
    "N2": "d6h",
    "Cl2": "d6h",
    "H2O2": "c2",
    "Na2": "d6h",
    "BeH": "c6v",
    "C3H4_C2v": "c2v",
    "NO2": "c2v",
}

spin_polarized = [
    "H2COH",
    "C3H9C",
    "SiH2_s3B1d",
    "CH3CO",
    "HCO",
    "Si",
    "CN",
    "S",
    "F",
    "C2H3",
    "PH2",
    "SiH3",
    "NO",
    "NH2",
    "CH",
    "C",
    "CH3S",
    "OH",
    "Cl",
    "Al",
    "CH3O",
    "Na",
    "CH3CH2O",
    "C3H7",
    "CH3",
    "P",
    "S2",
    "H",
    "C2H5",
    "B",
    "ClO",
    "O2",
    "SO",
    "NH",
    "CH2_s3B1d",
    "O",
    "CCH",
    "N",
    "Si2",
    "SH",
    "Li",
    "BeH",
    "NO2",
]

for mol in spin_polarized:
    del systems[mol]

# Have just one representative of the each symmetry group
groups = {v: k for k, v in systems.items()}
systems = {v: k for k, v in groups.items()}

assert set(systems.values()) == {
    "d2d",
    "d3d",
    "d6h",
    "c2",
    "cs",
    "td",
    "c2v",
    "c6v",
    "d3h",
    "c3v",
    "oh",
    "d2h",
    "c2h",
    "c1",
}


# P2 make cell hexagonal
def build_cell(atoms):
    if len(atoms) == 2:
        L = 8
        angle = 2 * pi / 3
        c, s = cos(angle), sin(angle)
        cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
        atoms.set_cell(cell)
        atoms.set_pbc((True, True, True))
        atoms.center()
    else:
        atoms.center(vacuum=3)


@pytest.mark.parametrize("name,symmetry", systems.items())  # g2.names
def test_molecule(name, symmetry):
    atoms = molecule(name)
    build_cell(atoms)
    with workdir(name):
        write(name + ".xyz", atoms)
    if 0:
        os.system(f"rm -r {name}")
        Path(name).mkdir(existok=True)
        with workdir(name):
            os.system(f"x2t {name}.xyz > coord")
            Path("inp").write_text(turbomole_input)
            os.system("define < inp > define_out.txt")
            os.system("dscf >output.txt")
            os.system(
                'cat output.txt |grep "irrep      " --after=3 --no-group-separator > irreps.txt'
            )
            os.system(
                'cat output.txt |grep "symmetry group of the molecule :" > group.txt'
            )
    with workdir(name):
        states = parse_eigenvalues(Path("irreps.txt").read_text())
        tmole_group = Path("group.txt").read_text().split()[-1]
    assert tmole_group.upper() == symmetry.upper()
    for state in states:
        print(f"{state}")

    with workdir(name):
        if not Path("wfs.gpw").exists():
            calc = GPAW(mode={"name": "pw"}, xc="PBE")
            atoms.calc = calc
            atoms.get_potential_energy()
            calc.write("wfs.gpw", mode="all")

    with workdir(name):
        calc = GPAW("wfs.gpw")
        results = analyze_symmetry(calc, False, expected=symmetry)
    print(results)
    print(states)


if __name__ == "__main__":
    print(set(systems.values()))
    for system in g2.names:
        group = "???"
        try:
            group = Path(system + "/group.txt").read_text().split()[-1]
        except FileNotFoundError:
            pass
        print(f'"{system}": "{group}",')
