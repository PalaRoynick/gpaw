import numpy as np


def pretty_header(header, log):
    bar = " " + "═" * (len(header) + 14)
    log(bar)
    log("     " + header)
    log(bar)
    log()


def pretty_subheader(header, log):
    bar = " " * 2 + "─" * (len(header) + 10)
    log(bar)
    log(" " * 4 + header)
    log(bar)


def atos(array, fmt):
    return " ".join(f"{x:{fmt}}" for x in array)


def pretty(C_cv, title=None, units=None, decimals=7, symbolize=False, eps=1e-4, *, log):
    print('Pretty getting', C_cv, title)
    C_cv = C_cv.copy()
    if symbolize:
        # Find smallest non zero
        alpha = np.min(np.abs(C_cv[np.nonzero(np.abs(C_cv) > eps)]))
        C_cv = C_cv / alpha
        if np.allclose(C_cv - np.round(C_cv), 0, atol=1e-3):
            C_cv = np.round(C_cv)
            decimals = 0
    else:
        C_cv = np.round(C_cv, decimals=decimals)

    # Remove signed zero
    C_cv = np.where(C_cv == 0, 0.0, C_cv)

    if title:
        log(f"{title} [{units}]")
    for i in range(C_cv.shape[0]):
        for j in range(C_cv.shape[1]):
            log(f"{C_cv[i, j]:{decimals + 5}.{decimals}f} ", end="")
        log()


def pprint_atoms(atoms, log, units="Å"):
    cell = atoms.cell
    log(f"Unit cell ({units})")
    for i in range(3):
        log(f" a{i + 1} = [ ", end="")
        for j in range(3):
            log(f" {cell[i, j]:10.5f}", end="")
        log(" ]")
    log(f"Lengths ({units}): {atos(cell.lengths(), '.3f')}")
    log(f"Angles (°): {atos(cell.angles(), '.2f')}")
    log("Atoms:")
    atom_table(atoms, log=log)

def pretty_atomic_dofs(atoms, dof_zac, *, log):
    C_cv = atoms.cell
    log("Atomic degrees of freedom:")

    from dataclasses import dataclass

    @dataclass
    class FakeAtom:
        index: int
        symbol: str
        position: np.ndarray
        scaled_position: np.ndarray

    for z, dof_ac in enumerate(dof_zac):
        log(f'Degree of freedom q{z:02d}')
        from ase import Atom
        atoms = [FakeAtom(atom.index,
                          atom.symbol,
                          dof_ac[atom.index] @ C_cv,
                          dof_ac[atom.index]) for atom in atoms]
        atom_table(atoms, log=log)
        log()

def atom_table(atoms, *, log):    
    log('   id symbol  Rx         Ry         Rz         sx         sy         sz')
    for a in atoms:
        s = f"{a.index:5d} {a.symbol:5s}"
        for v in range(3):
            s += f"{a.position[v]:10.5f} "
        for v in range(3):
            s += f"{a.scaled_position[v]:20.15f} "
        log(s)


def pretty_dofs(dM_zcc, M_cc, rot_vv, C_cv, eps=1e-8, *, log):
    from gpaw.new.relax import chol_derivative

    log(f"Found {len(dM_zcc)} independent cell degrees of freedom")
    for z, dM_cc in enumerate(dM_zcc):
        log(f"Tangent {z} of cell")
        log("In metric space")
        pretty(dM_cc, symbolize=True, decimals=3, log=log)
        log("In cell space at C0_cv:")
        dC_cv = chol_derivative(M_cc, dM_cc) @ rot_vv.T
        pretty(dC_cv, symbolize=True, decimals=6, log=log)

        log("In terms of unit cell vectors a1, a2, a3")
        # Deformation gradient, but in cc space
        F_cc = np.linalg.inv(C_cv.T) @ dC_cv.T
        pretty(F_cc, symbolize=True, decimals=3, log=log)
