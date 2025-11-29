from ase.collections import g2
import pytest
from ase.io import write
from ase.build import molecule
from pathlib import Path
from gpaw.utilities.pointgroup import PointGroup, Projectable, SPGOperations
from gpaw.utilities.pointgroup_data import character_tables

import os
import contextlib


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
    assert spg_ops.pointgroup == expected
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
from dataclasses import dataclass


@dataclass
class State:
    irrep: str
    eigenvalue: float
    occupation: float
    degeneracy: int = 1

    def __format__(self, fmt):
        return f"{self.irrep:5s} {self.eigenvalue:8.2f} {self.occupation:5.2f}"


from itertools import zip_longest


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
            print("asd")
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


systems = {"H2O": "C2v", "P2": "D6h", "CO": "C6v", "NaCl": "C6v", "H2": "D6h"}

systems = {
    "PH3": "???",
    "P2": "???",
    "CH3CHO": "???",
    "H2COH": "???",
    "CS": "???",
    "OCHCHO": "???",
    "C3H9C": "???",
    "CH3COF": "???",
    "CH3CH2OCH3": "???",
    "HCOOH": "???",
    "HCCl3": "???",
    "HOCl": "???",
    "H2": "???",
    "SH2": "???",
    "C2H2": "???",
    "C4H4NH": "???",
    "CH3SCH3": "???",
    "SiH2_s3B1d": "???",
    "CH3SH": "???",
    "CH3CO": "???",
    "CO": "???",
    "ClF3": "???",
    "SiH4": "???",
    "C2H6CHOH": "???",
    "CH2NHCH2": "???",
    "isobutene": "???",
    "HCO": "???",
    "bicyclobutane": "???",
    "LiF": "???",
    "Si": "???",
    "C2H6": "???",
    "CN": "???",
    "ClNO": "???",
    "S": "???",
    "SiF4": "???",
    "H3CNH2": "???",
    "methylenecyclopropane": "???",
    "CH3CH2OH": "???",
    "F": "???",
    "NaCl": "???",
    "CH3Cl": "???",
    "CH3SiH3": "???",
    "AlF3": "???",
    "C2H3": "???",
    "ClF": "???",
    "PF3": "???",
    "PH2": "???",
    "CH3CN": "???",
    "cyclobutene": "???",
    "CH3ONO": "???",
    "SiH3": "???",
    "C3H6_D3h": "???",
    "CO2": "???",
    "NO": "???",
    "trans-butane": "???",
    "H2CCHCl": "???",
    "LiH": "???",
    "NH2": "???",
    "CH": "???",
    "CH2OCH2": "???",
    "C6H6": "???",
    "CH3CONH2": "???",
    "cyclobutane": "???",
    "H2CCHCN": "???",
    "butadiene": "???",
    "C": "???",
    "H2CO": "???",
    "CH3COOH": "???",
    "HCF3": "???",
    "CH3S": "???",
    "CS2": "???",
    "SiH2_s1A1d": "???",
    "C4H4S": "???",
    "N2H4": "???",
    "OH": "???",
    "CH3OCH3": "???",
    "C5H5N": "???",
    "H2O": "???",
    "HCl": "???",
    "CH2_s1A1d": "???",
    "CH3CH2SH": "???",
    "CH3NO2": "???",
    "Cl": "???",
    "Be": "???",
    "BCl3": "???",
    "C4H4O": "???",
    "Al": "???",
    "CH3O": "???",
    "CH3OH": "???",
    "C3H7Cl": "???",
    "isobutane": "???",
    "Na": "???",
    "CCl4": "???",
    "CH3CH2O": "???",
    "H2CCHF": "???",
    "C3H7": "???",
    "CH3": "???",
    "O3": "???",
    "P": "???",
    "C2H4": "???",
    "NCCN": "???",
    "S2": "???",
    "AlCl3": "???",
    "SiCl4": "???",
    "SiO": "???",
    "C3H4_D2d": "???",
    "H": "???",
    "COF2": "???",
    "2-butyne": "???",
    "C2H5": "???",
    "BF3": "???",
    "N2O": "???",
    "F2O": "???",
    "SO2": "???",
    "H2CCl2": "???",
    "CF3CN": "???",
    "HCN": "???",
    "C2H6NH": "???",
    "OCS": "???",
    "B": "???",
    "ClO": "???",
    "C3H8": "???",
    "HF": "???",
    "O2": "???",
    "SO": "???",
    "NH": "???",
    "C2F4": "???",
    "NF3": "???",
    "CH2_s3B1d": "???",
    "CH3CH2Cl": "???",
    "CH3COCl": "???",
    "NH3": "???",
    "C3H9N": "???",
    "CF4": "???",
    "C3H6_Cs": "???",
    "Si2H6": "???",
    "HCOOCH3": "???",
    "O": "???",
    "CCH": "???",
    "N": "???",
    "Si2": "???",
    "C2H6SO": "???",
    "C5H8": "???",
    "H2CF2": "???",
    "Li2": "???",
    "CH2SCH2": "???",
    "C2Cl4": "???",
    "C3H4_C3v": "???",
    "CH3COCH3": "???",
    "F2": "???",
    "CH4": "???",
    "SH": "???",
    "H2CCO": "???",
    "CH3CH2NH2": "???",
    "Li": "???",
    "N2": "???",
    "Cl2": "???",
    "H2O2": "???",
    "Na2": "???",
    "BeH": "???",
    "C3H4_C2v": "???",
    "NO2": "???",
}

# P2 make cell hexagonal


@pytest.mark.parametrize("name,symmetry", systems.items())  # g2.names
def test_molecule(name, symmetry):
    atoms = molecule(name)
    # os.system(f'rm -r {name}')
    Path(name).mkdir(exist_ok=True)
    with workdir(name):
        if 1:
            write(name + ".xyz", atoms)
            os.system(f"x2t {name}.xyz > coord")
            Path("inp").write_text(turbomole_input)
            os.system("define < inp")
            os.system("dscf >output.txt")
            os.system(
                'cat output.txt |grep "irrep      " --after=3 --no-group-separator > irreps.txt'
            )
            os.system(
                'cat output.txt |grep "symmetry group of the molecule :" > group.txt'
            )
        states = parse_eigenvalues(Path("irreps.txt").read_text())
        tmole_group = Path("group.txt").read_text().split()[-1]
        assert tmole_group.upper() == symmetry.upper()
        for state in states:
            print(f"{state}")
    return
    from gpaw.new.ase_interface import GPAW

    calc = GPAW(mode={"name": "pw"}, xc="PBE")
    atoms.center(vacuum=4)
    atoms.calc = calc
    atoms.get_potential_energy()
    results = analyze_symmetry(calc, False, expected=symmetry)
    print(results)
    print(states)


print(g2.names)
for system in g2.names:
    print(f'"{system}": "???",')
