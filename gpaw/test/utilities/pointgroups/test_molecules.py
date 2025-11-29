import numpy as np
from ase.collections import g2
import pytest
from ase.io import write
from ase.build import molecule
from pathlib import Path
from gpaw.utilities.pointgroup import PointGroup, Projectable, SPGOperations
from gpaw.new.ase_interface import GPAW
from dataclasses import dataclass
from numpy import pi, sin, cos
import re
import json
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


@dataclass
class State:
    irrep: str
    eigenvalue: float
    occupation: float
    degeneracy: int = 1
    weight: float = 1.0

    def __format__(self, fmt):
        return f"{self.irrep:5s} {self.eigenvalue:8.2f} {self.occupation:5.2f}"

    def new(self, irrep=None, eigenvalue=None, occupation=None, degeneracy=None, weight=None):
        return State(irrep or self.irrep, eigenvalue or self.eigenvalue, occupation or self.occupation, degeneracy or self.degeneracy, weight or self.weight)

    def as_dict(self):
        return {
            "irrep": self.irrep,
            "eigenvalue": self.eigenvalue,
            "occupation": self.occupation,
            "degeneracy": self.degeneracy,
            "weight": self.weight,
        }


@dataclass
class SymmetryEigenvalues:
    little_group: str
    states: list[State]

    def __post_init__(self):
        if isinstance(self.states[0], dict):
            self.states = [State(**state) for state in self.states]

    def save(self, filename: str):
        Path(filename).write_text(json.dumps(self.as_dict()))

    def as_dict(self):
        return {"little_group": self.little_group, "states": [state.as_dict() for state in self.states]}

    @classmethod
    def load(cls, filename: str):
        return SymmetryEigenvalues(**json.loads(Path(filename).read_text()))

    @classmethod
    def from_calc(cls, calc, layergroup=False):
        spg_ops = SPGOperations.from_atoms(calc.atoms, layergroup=layergroup)
        pg = PointGroup(spg_ops, [0, 0, 1]) # if layergroup else None)
        states = []
        failure = False
        eig_n = calc.get_eigenvalues()
        occ_n = calc.get_occupation_numbers()
        for band, (eig, occ) in enumerate(zip(eig_n, occ_n)):
            signature = pg.signature(Projectable.from_calc(calc, band))
            found = None
            for irrep, s in zip(
                pg.character_table.irreps,
                pg.detect_irrep(signature),
            ):
                if s > 0.01:
                    print(band, irrep, f"{s.real:.2f}")
                    states.append(State(irrep, eig, occ, 1, s))
                    if found is not None:
                        if occ > 1e-2:
                            failure = True
                    found = irrep
        if failure:
            raise ValueError("Band spans multiple irreps.")
        return cls(spg_ops.pointgroup, states)

    def __isub__(self, value):
        if isinstance(value, float):
            for state in self.states:
                state.eigenvalue -= value
            return self
        else:
            raise TypeError("Cannot subtract {value}")

    def unroll_degeneracies(self):
        states = []
        for state in self.states:
            for _ in range(state.degeneracy):
                states.append(state.new(degeneracy=1))
        return SymmetryEigenvalues(self.little_group, states)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.states[item]
        if isinstance(item, slice):
            return SymmetryEigenvalues(self.little_group, self.states[item])
        raise NotImplementedError

    def __repr__(self):
        s = f"Point group: {self.little_group}"
        s += "\n"
        for state in self.states:
            s += f"{state}\n"
        return s

    @property
    def occupations(self):
        return [state.occupation for state in self.states]

    @property
    def irreps(self):
        return [state.irrep for state in self.states]

    @property
    def occupied_states(self):
        HOMO = np.where(self.occupations)[0][-1]
        return SymmetryEigenvalues(self.little_group, self.states[: HOMO + 1])

    def __len__(self):
        return np.sum([state.degeneracy for state in self.states])


turbomole_input = """

a coord
desy 1e-3
*
no
b
all def2-TZVPP
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
                # Remove the leading eigenvalue index
                irr = re.sub(r"^\d+", "", irr).upper()
                irr = irr.replace('"', "''")
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
    "C2H6SO": "cs",
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

ready = {'c1', 'ci', 'cs', 'c2', 'c2h', 'c2v'}
systems = {key: value for key, value in systems.items() if value in ready}


def build_cell(atoms, group):
    if group.upper() in {'D3D', 'D6H'}:
        L = 10
        angle = 2 * pi / 3
        c, s = cos(angle), sin(angle)
        cell = [[L, 0, 0], [c * L, s * L, 0], [0, 0, L]]
        atoms.set_cell(cell)
        atoms.set_pbc((True, True, True))
        atoms.center()
    else:
        atoms.center(vacuum=6)


@pytest.mark.parametrize("name,symmetry", systems.items())  # g2.names
def test_molecule(name, symmetry):
    atoms = molecule(name)
    build_cell(atoms, symmetry)
    tmole_json = Path(name + "_tmole.json")
    if 1: #not tmole_json.exists():
        os.system(f"rm -r {name}")
        Path(name).mkdir(exist_ok=True)
        with workdir(name):
            write(name + ".xyz", atoms)
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
            states = parse_eigenvalues(Path("irreps.txt").read_text())
            tmole_group = Path("group.txt").read_text().split()[-1]
            assert tmole_group.upper() == symmetry.upper()
            tmole_states = SymmetryEigenvalues(tmole_group, states)
        tmole_states.save(tmole_json)
    tmole_states = SymmetryEigenvalues.load(tmole_json)

    with workdir(name):
        if not Path("wfs.gpw").exists():
            calc = GPAW(
                mode={"name": "pw", "ecut": 400, "force_complex_dtype": True}, xc="PBE",
                txt="gpaw.txt",
            )
            atoms.set_pbc((False, False, False))
            atoms.calc = calc
            atoms.get_potential_energy()
            calc.write("wfs.gpw", mode="all")

    with workdir(name):
        calc = GPAW("wfs.gpw")
        gpaw_states = SymmetryEigenvalues.from_calc(calc, False)
        assert gpaw_states.little_group.upper() == tmole_states.little_group.upper()
    print(f'{tmole_states}\n{gpaw_states}')
    gpaw_states = gpaw_states.occupied_states
    tmole_states = tmole_states.unroll_degeneracies().occupied_states
    comparable = min(len(gpaw_states), len(tmole_states))
    gpaw_states = gpaw_states[-comparable:]
    tmole_states = tmole_states[-comparable:]
    for tmole_state, gpaw_state in zip(tmole_states, gpaw_states):
        print(f"{tmole_state} | {gpaw_state}")
    for tmole_state, gpaw_state in zip(tmole_states, gpaw_states):
        assert tmole_state.irrep.upper() == gpaw_state.irrep.upper()
        assert np.abs(tmole_state.eigenvalue - gpaw_state.eigenvalue) < 0.4


if __name__ == "__main__":
    print(set(systems.values()))
    for system in g2.names:
        group = "???"
        try:
            group = Path(system + "/group.txt").read_text().split()[-1]
        except FileNotFoundError:
            pass
        print(f'"{system}": "{group}",')
