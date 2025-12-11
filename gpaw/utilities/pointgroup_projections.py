from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from gpaw.utilities.pointgroup import SPGOperations, PointGroup
from gpaw.utilities.pointgroup_proj import Projectable, PaniProjectable

@dataclass
class State:
    irrep: str
    eigenvalue: float
    occupation: float
    degeneracy: int = 1
    weight: float = 1.0

    def __format__(self, fmt):
        return f"{self.irrep:5s} {self.eigenvalue:8.2f} {self.occupation:5.2f}"

    def new(
        self, irrep=None, eigenvalue=None, occupation=None, degeneracy=None, weight=None
    ):
        return State(
            irrep or self.irrep,
            eigenvalue or self.eigenvalue,
            occupation or self.occupation,
            degeneracy or self.degeneracy,
            weight or self.weight,
        )

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
        return {
            "little_group": self.little_group,
            "states": [state.as_dict() for state in self.states],
        }

    @classmethod
    def load(cls, filename: str):
        return SymmetryEigenvalues(**json.loads(Path(filename).read_text()))

    @classmethod
    def from_calc(cls, calc, layergroup=False, *, pani):
        spg_ops = SPGOperations.from_atoms(calc.atoms, layergroup=layergroup)
        for op_cc in calc.symmetry.op_scc:
            for W_cc in spg_ops.W_scc:
                if np.allclose(op_cc.T, W_cc):
                    break
            else:
                raise ValueError(f"Symmetry not found. {op_cc}.")

        pg = PointGroup(spg_ops)  # , [0, 0, 1])  # if layergroup else None)
        states = []
        failure = False
        eig_n = calc.get_eigenvalues()
        occ_n = calc.get_occupation_numbers()
        for band, (eig, occ) in enumerate(zip(eig_n, occ_n)):
            if pani:
                P_ai = {}
                P_ani = calc.wfs.kpt_u[0].P_ani
                for a in P_ani.keys():
                    P_ai[a] = P_ani[a][band]
                proj = PaniProjectable(P_ai, calc.atoms, calc.wfs.setups)
                signature = pg.signature(proj)

            else:
                signature = pg.signature(Projectable.from_calc(calc, band))
            found = None
            for irrep, s in zip(
                pg.character_table.irreps,
                pg.detect_irrep(signature),
            ):
                if s > 1e-5: # 0.01:
                    print(band, eig, occ, irrep, f"{s.real:.2f}")
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
        return np.array([state.occupation for state in self.states])

    @property
    def irreps(self):
        return [state.irrep for state in self.states]

    @property
    def occupied_states(self):
        HOMO = np.where(self.occupations > 0.01)[0][-1]
        return SymmetryEigenvalues(self.little_group, self.states[: HOMO + 1])

    def __len__(self):
        return np.sum([state.degeneracy for state in self.states])

