from dataclasses import dataclass
import numpy as np
from gpaw.utilities.pointgroup_operation import OperationInfo, ppstr
from gpaw.utilities.pointgroup_data import character_tables
from ase import Atoms

def debugprint(*args, **kwargs):
    pass


@dataclass
class CharacterTable:
    irreps: list[str]
    classes: list[str]
    characters_ig: np.array
    classes_textbook: list[str] | None = None

    def __post_init__(self):
        if self.classes_textbook is None:
            self.classes_textbook = self.classes

    def class_to_textbook(self, name):
        return self.classes_textbook[self.classes.index(name)]

    @property
    def order(self):
        E = self.classes.index("E")
        return np.sum(self.characters_ig[:, E] ** 2)

    @property
    def normalized_characters_ig(self):
        # TODO: Actually pass on ng's
        def extract_n(s):
            if s == "E":
                return 1
            return int(s[0])

        n_g = np.array([extract_n(cls) for cls in self.classes])
        return np.conjugate(self.characters_ig) * n_g[None, :] / self.order

    @property
    def normalized_characters_ig2(self):
        # TODO: Actually pass on ng's
        def extract_n(s):
            if s == "E":
                return 1
            return int(s[0])

        # n_g = np.array([extract_n(cls) for cls in self.classes])
        return self.characters_ig / self.characters_ig[:, :1]

    @classmethod
    def from_data(cls, *, irreps, classes, table, classes_textbook=None):
        return CharacterTable(irreps, classes, np.array(table), classes_textbook)

    def in_conjugacy_order(self, names_g):
        indices = [self.classes.index(name) for name in names_g]
        return CharacterTable(
            self.irreps,
            [self.classes[idx] for idx in indices],
            self.characters_ig[:, indices],
            [self.classes_textbook[idx] for idx in indices],
        )

    def detect_irrep(self, signature_g):
        return np, sum(self.normalized_characters_ig * signature_g[None, :], axis=1)

    def detect_irrep2(self, signature_g):
        return np.linalg.solve(self.normalized_characters_ig2.T, signature_g)

    def print(self):
        is_complex = np.any(np.iscomplex(self.characters_ig))
        symbols = [
            ("ε", np.exp(2j * np.pi / 3), "exp(2πi/3)"),
            ("ε*", np.exp(-2j * np.pi / 3), "exp(-2πi/3)"),
        ]
        used_values = []
        # if is_complex:
        #    columns = 12
        # else:
        #    columns = 7
        columns = 6

        def format_number(n):
            if np.isclose(n, np.round(n)):
                assert np.abs(n.imag) < 1e-5
                return "%+02d" % n.real
            return "%+.2f" % n

        def complex_format(n):
            if np.isclose(n, 0):
                return " 0"
            if np.isclose(n.imag, 0):
                return format_number(n)
            if np.isclose(n.real, 0):
                return format_number(n.imag) + "i"
            for used, (symbol, value, text) in enumerate(symbols):
                if np.isclose(value, n):
                    used_values.append(used)
                    return symbol
                if np.isclose(value, -n):
                    used_values.append(used)
                    return "-" + symbol
            return format_number(n.real) + format_number(n.imag) + "i"

        try:
            print("%-20s" % "irreps/classes", end="")

            for name in self.classes:
                print(f"%-{columns}s" % name, end="")
            print()
            for i, iname in enumerate(self.irreps):
                print("%-20s" % iname, end="")
                for g, gname in enumerate(self.classes):
                    chi = self.characters_ig[i, g]
                    chi_str = complex_format(chi)
                    print(f"%-{columns}s" % chi_str, end="")
                print()
            for value in set(used_values):
                symbol, value, text = symbols[value]
                print(f"{symbol} = {text} = {value}")
        except ValueError:
            print("...")


@dataclass
class SPGOperations:
    atoms: Atoms
    W_scc: np.array
    w_sc: np.array
    origin_shift_c: np.array
    cell_cv: np.array
    pointgroup: str
    allow_translations: bool = False

    def __post_init__(self):
        if not self.allow_translations:
            print(f'Before {self}')
            operations_now = len(self.w_sc)
            new_W_scc, new_w_sc = [], []
            for W_cc, w_c in zip(self.W_scc, self.w_sc):
                if np.allclose(w_c, 0, atol=0.01):
                    new_W_scc.append(W_cc)
                    new_w_sc.append(w_c)
            self.W_scc = new_W_scc
            self.w_sc = new_w_sc

            operations_after = len(self.w_sc)
            if operations_now != operations_after:
                import warnings
                raise ValueError(f'Translational operations removed'
                              f'{operations_now} -> {operations_after}.'
                              'TODO: make sure they were actually correctly '
                              'handled.')

            if not np.allclose(self.w_sc, 0):
                print(f"{self}")
                raise ValueError("Point groups don't support translations.")

    @property
    def character_table(self):
        from gpaw.utilities.pointgroup_data import character_tables

        return CharacterTable.from_data(**character_tables[self.pointgroup])

    @classmethod
    def from_atoms(cls, atoms, verbose=False, *, layergroup: bool, symprec: float = 1e-1):
        # XXX: NOTE! This will modify the atoms!!!
        #print("Modifying atoms with translations (for now, temporarily)")
        if layergroup:
            return cls.from_atoms_layergroup(atoms, verbose=verbose, symprec=symprec)
        else:
            return cls.from_atoms_spacegroup(atoms, verbose=verbose, symprec=symprec)

    @classmethod
    def from_atoms_spacegroup(cls, atoms, verbose=False, symprec=1e-1):
        if verbose:
            print("Detecting spacegroup (NOT layergroup) with spglib...")
        import spglib

        dataset = spglib.get_symmetry_dataset(
            cell=(
                atoms.get_cell(),
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers(),
            ),
            symprec=symprec,
        )
        debugprint(f"{dataset=}")
        print(f"{dataset=}")
        return cls.from_dataset(dataset, atoms, verbose=verbose)

    @classmethod
    def from_atoms_layergroup(cls, atoms, verbose=False, symprec=1e-1):
        if verbose:
            print("Detecting layergroup (NOT spacegroup) with spglib...")
        import spglib

        dataset = spglib.get_layergroup(
            cell=(
                atoms.get_cell(),
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers(),
            ),
            aperiodic_dir=2,
            symprec=symprec,
        )
        debugprint(f"{dataset=}")
        return cls.from_dataset(dataset, atoms, verbose=verbose)

    @classmethod
    def from_dataset(cls, dataset, atoms, verbose=False):
        W_scc = dataset.rotations
        w_sc = dataset.translations
        #origin_shift_c = dataset.origin_shift
        origin_shift_c = np.linalg.inv(dataset.transformation_matrix) @ dataset.origin_shift
        cell_cv = np.array(atoms.cell)
        # assert np.allclose(dataset.transformation_matrix, np.eye(3))
        from gpaw.utilities.pointgroup_data import spglib_to_schoenflies

        # Filter glide/screw operations: w' = w + (W - I) @ t is ≈ 0 for
        # pure point-group ops; non-zero for glides/screws.
        # See _shifted_translations.
        w_shifted = cls._shifted_translations(W_scc, w_sc, origin_shift_c)
        pure_mask = np.all(np.abs(w_shifted) < 0.01, axis=1)
        n_glides = int(np.sum(~pure_mask))

        if verbose:
            intl_before = spglib_to_schoenflies.get(
                dataset.pointgroup, dataset.pointgroup)
            print(f"  Before filtering: {len(W_scc)} operations, "
                  f"PG={intl_before} ({dataset.pointgroup}), "
                  f"origin_shift={origin_shift_c}",
                  f"W_scc: {W_scc}, w_sc: {w_sc}",
                  f"rot_vv: {cell_cv.T @ W_scc[0] @ np.linalg.inv(cell_cv).T}",)
            for s in range(len(W_scc)):
                info = OperationInfo.from_op(W_scc[s])
                tag = "PURE" if pure_mask[s] else "GLIDE/SCREW"
                print(f"    [{s:2d}] {tag:11s}  w={w_sc[s]}  "
                      f"w'={w_shifted[s]}  {info}")

        if n_glides > 0:
            import spglib as _spglib
            W_scc = W_scc[pure_mask]
            w_sc = w_sc[pure_mask]
            pg_info = _spglib.get_pointgroup(W_scc)
            intl_symbol = pg_info[0].strip()
            pointgroup = spglib_to_schoenflies.get(intl_symbol, "C1")
            if verbose:
                print(f"  After filtering: {len(W_scc)} operations kept, "
                      f"{n_glides} glide/screw removed; "
                      f"PG={pointgroup} ({intl_symbol})")
        else:
            pointgroup = spglib_to_schoenflies[dataset.pointgroup]
            if verbose:
                print(f"  No glides/screws; "
                      f"PG={pointgroup} ({dataset.pointgroup})")

        return cls(
            atoms,
            W_scc,
            w_sc,
            origin_shift_c,
            cell_cv,
            pointgroup,
            allow_translations=True,
        )

    @staticmethod
    def _shifted_translations(W_scc, w_sc, origin_shift_c):
        """Residual translations after applying the origin shift.

        w'_sc = w_sc + (W_scc - I) @ origin_shift_c
             = w_sc + W_scc @ origin_shift_c - origin_shift_c

        Pure point-group operations have w' ≈ 0 (mod 1); glides/screws
        retain a non-zero fractional translation.
        """
        w_shifted = (w_sc
                     + np.einsum("scd,d->sc", W_scc, origin_shift_c)
                     - origin_shift_c)
        # Round and wrap to [-0.5, 0.5) so values near 0 stay near 0
        w_shifted = (np.round(w_shifted * 100) / 100 + 0.5) % 1.0 - 0.5
        return w_shifted

    @property
    def O_svv(self):
        return np.array(
            [
                self.cell_cv.T @ W_cc @ np.linalg.inv(self.cell_cv).T
                for W_cc in self.W_scc
            ]
        )

    def __repr__(self):
        s = ""
        for W_cc, w_c in zip(self.W_scc, self.w_sc):
            s += ppstr(W_cc, w_c)
        return s


class SymmmetryOperations:
    def __init__(self, ops_occ):
        self.ops_occ = ops_occ
        self.inv_oo = None
        self.mul_oo = None

        self._build_multiplication_table()

        self.operation_info_o = [OperationInfo.from_op(op_cc) for op_cc in ops_occ]

    def get_op_id(self, op_cc):
        for o1, op1_cc in enumerate(self.ops_occ):
            if np.linalg.norm(op_cc - op1_cc) < 0.01:
                return o1
        raise ValueError("Unknown operation: %s." % str(op_cc))

    def _build_multiplication_table(self):
        ops_occ = self.ops_occ
        N = len(ops_occ)
        mul_oo = np.zeros((N, N), dtype=int)
        inv_oo = np.zeros((N, N), dtype=int)
        for o1, op1_cc in enumerate(ops_occ):
            for o2, op2_cc in enumerate(ops_occ):
                o3 = self.get_op_id(np.dot(op1_cc, op2_cc))
                mul_oo[o1, o2] = o3
                inv_oo[o1, o3] = o2
        self.mul_oo, self.inv_oo = mul_oo, inv_oo


class ConjugacyClassClassifierClass:
    def __init__(
        self,
        operations: SymmmetryOperations,
        expected_classes: list[str],
        verbose=True,
        atoms=None,
    ):
        self.operations = operations
        self.expected_classes = expected_classes
        self.principal_axis = "Principal axis not yet detected."
        self.atoms = atoms
        self.axes = {}
        ops_occ = operations.ops_occ
        N = len(ops_occ)
        op_pool = range(N)
        self.ops_g = []

        self.g_o = np.zeros((N,), int)  # Conjugacy class index for each op
        iterations = 0
        while True:
            iterations += 1
            if iterations >= 1_000:
                breakpoint()
                raise RuntimeError('Unexpected infinite loop')
            # Loop until all operations are assigned a class
            if len(op_pool) == 0:
                break
            # Take an element from operation pool...
            op1 = op_pool[0]
            op1_cc = ops_occ[op1]
            # ...conjugate it with all possible operations, see the result, and remove any duplicates
            conjugacy_class = np.unique(
                [
                    operations.get_op_id(np.dot(op2_cc, np.dot(op1_cc, op2_cc.T)))
                    for o2, op2_cc in enumerate(ops_occ)
                ]
            )

            # Fill g_o array that maps ops to class
            for op in conjugacy_class:
                self.g_o[op] = len(self.ops_g)
            self.ops_g.append(conjugacy_class)
            # Remove all operations already assigned a class from the pool
            op_pool = list(set(op_pool) - set(conjugacy_class))

        if verbose:
            print("Found %d conjugacy classes" % len(self.ops_g))

    def _detect_main_conjugacy_class(self, operation_info_o):
        """Detect the main conjugacy class

        Essentially, either E, i, or a number indicating the size of the conjugacy
        class and then operation. However, at this point, one does not yet
        distinguish different types of reflections.
        """
        N = len(operation_info_o)
        first_info = operation_info_o[0]

        # Special cases (omit 1 in front)
        if len(operation_info_o) == 1:
            if first_info.identity:
                return "E"
            if first_info.inversion:
                return "i"

        if all([info.cls == first_info.cls for info in operation_info_o]):
            return f"{N}{first_info.cls}"

        debugprint(f"{operation_info_o=}")
        raise ValueError("Could not detect conjugacy class.")

    def _resolve_1sv(self, count, operations):
        if count != 2:
            raise NotImplementedError(f"1sv count {count}")
        axes = [operations[0].axis, operations[1].axis]
        debugprint(f"DEBUG: _resolve_1sv axes={axes}")
        debugprint(f"DEBUG: self.axes={self.axes}")

        # If axes are already defined (e.g. by 1C2), use them
        if 'x' in self.axes and 'y' in self.axes:
            suffixes = [None, None]
            for i, axis in enumerate(axes):
                if np.isclose(abs(np.dot(axis, self.axes['x'])), 1.0):
                    suffixes[i] = "_yz"  # Normal x -> yz plane
                elif np.isclose(abs(np.dot(axis, self.axes['y'])), 1.0):
                    suffixes[i] = "_xz"  # Normal y -> xz plane
                else:
                    debugprint(f"DEBUG: axis {axis} not matching x {self.axes['x']} or y {self.axes['y']}")
            
            debugprint(f"DEBUG: suffixes={suffixes}")
            if all(suffixes):
                return suffixes
            # If we failed to match, fall through to other methods

        # Check for flat plane (C2v convention)
        # xz is the flat plane (containing atoms). Normal is y.
        is_flat_normal = [False, False]
        if self.atoms is not None:
            center = self.atoms.get_center_of_mass()
            pos = self.atoms.positions - center
            for i, axis in enumerate(axes):
                # Check if all atoms are in the plane perpendicular to axis
                projections = np.dot(pos, axis)
                if np.allclose(projections, 0, atol=0.1):
                    is_flat_normal[i] = True

        if sum(is_flat_normal) == 1:
            # Found exactly one flat plane. Label it _xz (normal y).
            # The other is _yz (normal x).
            return ["_xz" if is_flat else "_yz" for is_flat in is_flat_normal]

        if np.isclose(
            np.linalg.det([self.principal_axis, axes[0], axes[1]]), 1.0
        ):
            return ["_yz", "_xz"]
            # return ["_yz", "_xz"] # XXX This one works for C3H4
        elif np.isclose(
            np.linalg.det([self.principal_axis, axes[1], axes[0]]), 1.0
        ):
            return ["_xz", "_yz"]
            # return ["_xz", "_yz"] XXX This one works for C3H4
        else:
            print(self.principal_axis, axes)
            raise ValueError("Could not determine coordinate system.")

    def _resolve_1C2(self, count, operations):
        if count == 3:
            suffixes = [None] * 3

            # Identify z-axis (parallel to principal axis)
            z_idx = -1
            for i, op in enumerate(operations):
                if np.isclose(abs(np.dot(op.axis, self.principal_axis)), 1.0):
                    z_idx = i
                    break

            if z_idx == -1:
                 raise ValueError("Could not identify principal axis among 1C2 axes")

            suffixes[z_idx] = "_z"
            self.axes['z'] = operations[z_idx].axis

            # Identify x and y using handedness
            others = [i for i in range(3) if i != z_idx]
            idx1, idx2 = others

            det = np.linalg.det(
                [self.principal_axis, operations[idx1].axis, operations[idx2].axis]
            )
            
            debugprint(f"DEBUG: _resolve_1C2 principal={self.principal_axis} axes={[op.axis for op in operations]} det={det}")

            if np.isclose(det, 1.0):
                suffixes[idx1] = "_x"
                suffixes[idx2] = "_y"
                self.axes['x'] = operations[idx1].axis
                self.axes['y'] = operations[idx2].axis
            elif np.isclose(det, -1.0):
                suffixes[idx1] = "_y"
                suffixes[idx2] = "_x"
                self.axes['y'] = operations[idx1].axis
                self.axes['x'] = operations[idx2].axis
            else:
                raise ValueError(f"Determinant not +/- 1: {det}")
            
            debugprint(f"DEBUG: _resolve_1C2 suffixes={suffixes}")

            return suffixes
        raise NotImplementedError(f"1C2 count {count}")

    def _resolve_1C2_prime(self, count, operations):
        if count != 2:
            raise NotImplementedError(f"1C2' count {count}")
        axes = [operations[0].axis, operations[1].axis]
        det = np.linalg.det([self.principal_axis, axes[0], axes[1]])
        debugprint("DET", det, axes, self.principal_axis)
        if np.isclose(det, 1.0):
            return ["_y", "_x"]
        elif np.isclose(
            np.linalg.det([self.principal_axis, axes[1], axes[0]]), 1.0
        ):
            return ["_x", "_y"]
        else:
            debugprint(self.principal_axis, axes)
            raise ValueError("Could not determine coordinate system.")

    def _resolve_cyclic(self, name, count, operations):
        if count != 2:
            raise NotImplementedError(f"{name} count {count}")
        # Horrible code, refactor
        order = int(name[-1])
        if name[1] == "S" and order % 2 == 1:
            order *= 2
        odd = order - 1
        if operations[0].is_clockwise(self.principal_axis) and not operations[
            1
        ].is_clockwise(self.principal_axis):
            return ["", f"^{odd}"]
        elif not operations[0].is_clockwise(self.principal_axis) and operations[
            1
        ].is_clockwise(self.principal_axis):
            return [f"^{odd}", ""]
        else:
            raise ValueError("Cannot figure out.")

    def _resolve_2sv(self, count, operations):
        if count != 2:
            raise NotImplementedError(f"2sv count {count}")
        from numpy.linalg import norm

        cosines = []
        for cell_v in self.atoms.cell:
            cosines.append(np.dot(operations[0].axis, cell_v) / norm(cell_v))
        cosines = np.array(cosines)
        if np.allclose(cosines, np.round(cosines)):
            return ["-2sv", "-2sd"]
        else:
            return ["-2sd", "-2sv"]

    def _resolve_2C2_prime(self, count, operations):
        if count != 2:
            raise NotImplementedError(f"2C2' count {count}")
        #raise NotImplementedError
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX resolve_C2C_prime needs to be implemented')
        return ["", "'"]

    def _resolve_3C2_prime(self, count, operations):
        if count != 2:
            raise NotImplementedError(f"3C2' count {count}")
        raise NotImplementedError
        return ["", "'"]

    def _resolve_3sv(self, count, operations):
        if count != 2:
            raise NotImplementedError(f"3sv count {count}")
        raise NotImplementedError
        return ["-3sv", "-3sd"]

    def _detect_conjugacy_classes(self, class_names: list[str]):
        free_names = set(class_names)

        names_g = []

        main_conjugacy_classes = []
        # For each conjugacy class
        for g, classops in enumerate(self.ops_g):
            # There are the operations of current conjucagy class
            # ops_occ = [self.operations.ops_occ[o] for o in classops]
            operation_info_o = [self.operations.operation_info_o[o] for o in classops]

            # Favour C2 over sigma h
            # XXX: Consolidate this hack
            rotation_order = (operation_info_o[0].N or 0) + 0.1 * operation_info_o[
                0
            ].rotation
            main_conjugacy_classes.append(
                (
                    g,
                    self._detect_main_conjugacy_class(operation_info_o),
                    operation_info_o,
                    rotation_order,
                )
            )

        # Find the principal axis
        axis_determining_cc = max(main_conjugacy_classes, key=lambda x: x[3])
        principal_axis = axis_determining_cc[2][0].axis
        # print(f'{principal_axis=} from {axis_determining_cc[2][0]=}')
        if principal_axis is not None:
            assert np.linalg.norm(principal_axis.imag) < 1e-5

        # principal_axis = np.array([1,0,0])

        self.principal_axis = principal_axis
        names_g = []
        for g, main_cc, operation_info_o, rotation_order in main_conjugacy_classes:
            info = operation_info_o[0]
            if info.rotation:
                # Rotation orthogonal to the main axis, add a prime
                if np.allclose(np.dot(info.axis, principal_axis), 0):
                    if f"{main_cc}_z" in self.expected_classes:
                        debugprint(f"Skipping prime for {main_cc} because {main_cc}_z is expected")
                    else:
                        debugprint("Adding prime")
                        main_cc += "'"
            if info.reflection:
                if principal_axis is None:
                    names_g.append(main_cc)
                    continue
                # print('Analyzing reflection conjugacy class. Reflection planes:')
                Ds = []
                for info in operation_info_o:
                    assert info.reflection
                    Ds.append(np.dot(info.axis, principal_axis))
                    # print(info.op_cc)
                    # print(info.axis, 'D=', np.dot(info.axis, principal_axis))
                if np.allclose(Ds, 1.0):
                    main_cc += "h"
                elif np.allclose(Ds, 0.0):
                    main_cc += "v"  # XXX Might also be d sometimes
                else:
                    main_cc += "d"
            names_g.append(main_cc)

        # Find duplicate main classes to further distinguish them
        from collections import Counter

        duplicates = [(k, v) for k, v in Counter(names_g).items() if v > 1]
        duplicates.sort(key=lambda x: x[0])  # Ensure 1C2 comes before 1sv

        for name, count in duplicates:
            # Collect all of the axes of the conjugacy classes
            operations = []
            for g, _, operation_info_o, _ in main_conjugacy_classes:
                if names_g[g] != name:
                    continue
                # Getting information only from the first item
                o = operation_info_o[0]
                operations.append(o)
            if name == "1sv":
                extras = self._resolve_1sv(count, operations)
            elif name == "1C2":
                extras = self._resolve_1C2(count, operations)
            elif name == "1C2'":
                extras = self._resolve_1C2_prime(count, operations)
            elif name in {"1C3", "1S3", "1S4", "1C4", "1S6", "1C6"}:
                extras = self._resolve_cyclic(name, count, operations)
            elif name == "2sv":
                extras = self._resolve_2sv(count, operations)
            elif name == "2C2'":
                extras = self._resolve_2C2_prime(count, operations)
            elif name == "3C2'":
                extras = self._resolve_3C2_prime(count, operations)
            elif name == "3sv":
                extras = self._resolve_3sv(count, operations)
            else:
                raise NotImplementedError(
                    f"Duplicate conjugacy class name {name} count: {count}"
                )

            dpl_idx = [i for i, x in enumerate(names_g) if x == name]
            for idx, extra in zip(dpl_idx, extras):
                if extra and extra[0] == "-":
                    names_g[idx] = extra[1:]
                else:
                    names_g[idx] += extra
        """
        # We still might have two 1sv's
        # Hack for C2v
        one_es_vees = [g for g, name in enumerate(names_g) if name == '1sv']
        if len(one_es_vees) == 2:
            for g, add in zip(one_es_vees, ["_xz", "_yz"]):
                # TODO: Actually use orientation (maybe crystal axes)
                names_g[g] += add
       
        # Hack for D2
        one_es_vees = [g for g, name in enumerate(names_g) if name == '1C2']
        if len(one_es_vees) == 3:
            for g, add in zip(one_es_vees, ["_x", "_y", "_z"]):
                # TODO: Actually use some orientation
                names_g[g] += add
        """
        return names_g
        asd
        """
            if suggestion in free_names:
                names_g.append(suggestion)
                free_names.remove(suggestion)
            else:
                for free_name in free_names:
                    if free_name.startswith(suggestion):
                        names_g.append(free_name)
                        free_names.remove(free_name)
                        break
                else:
                    print('Conjugacy class:')
                    for info in operation_info_o:
                        print(ppstr(info.op_cc))
                        print(info)
                    raise ValueError(
                        f"Got unexpected conjugacy class {suggestion} free names: {free_names} all_names {class_names}"
                    )
        return names_g
        """

    @property
    def names_g(self):
        return self._detect_conjugacy_classes(self.expected_classes)


class PointGroup:
    def __init__(self, spg_ops):  # , principal_axis=[0, 0, 1]):
        # TODO: Create a class method to create from atoms
        # Make spg_ops obsolete
        self.spg_ops = spg_ops
        print("POINTGROUP", spg_ops.pointgroup)
        self.verbose = True
        # self.principal_axis = principal_axis

        self.operations = SymmmetryOperations(self.ops_occ)

        # self._build_multiplication_table()
        character_table = self.spg_ops.character_table

        self.c4 = ConjugacyClassClassifierClass(
            self.operations,
            expected_classes=character_table.classes,
            atoms=spg_ops.atoms,
            # principal_axis=principal_axis,
        )

        assert self.detected_pointgroup == self.spg_ops.pointgroup
        # self._find_conjugacy_classes()
        # self._build_character_table()

        # from collections import defaultdict
        # self.used_class_names = defaultdict(int)
        # self._detect_conjugacy_classes(class_names=character_table.classes)

        # self._detect_irreps()
        # self._name_groups_and_classes()
        # print(self.ops_g)
        # print(self.names_g)
        # self.print_character_table()

        if set(character_table.classes) != set(self.c4.names_g):
            print("Character tables classes", character_table.classes)
            print("Our names_g", self.c4.names_g)
            print(
                "Not in our names_g",
                set(character_table.classes) - set(self.c4.names_g),
            )
            print(
                "Not in our character_table",
                set(self.names_g) - set(character_table.classes),
            )
            raise ValueError("Cannot detect all of the conjugacy classes.")

        character_table = character_table.in_conjugacy_order(self.c4.names_g)
        character_table.print()

        # for i in range(self.character_ig.shape[0]):
        #    print(character_table.detect_irrep(signature_g=self.character_ig[i]))

        self.character_table = character_table

    @property
    def detected_pointgroup(self):
        names_g = set(self.c4.names_g)
        for name, table_data in character_tables.items():
            if set(table_data["classes"]) == names_g:
                print(f'Detected point group {name} from irreps {names_g}')
                return name
        else:
            raise ValueError(f'Cannot detect point group for conjugacy classes {names_g}')

    def signature(self, projectable):
        signature = np.zeros((len(self.names_g),), dtype=complex)
        
        # All projectables use fractional coords (op_cc, w_c) from W_scc
        for o, (op_cc, w_c) in enumerate(zip(self.spg_ops.W_scc, self.spg_ops.w_sc)):
            g = self.c4.g_o[o]
            signature[g] += (
                1
                / len(self.c4.ops_g[g])
                * projectable.dot(projectable.operation(op_cc, w_c=w_c))
            )
        return signature

    @property
    def names_g(self):
        return self.c4.names_g

    @property
    def textbook_names_g(self):
        return [
            self.spg_ops.character_table.class_to_textbook(name)
            for name in self.c4.names_g
        ]

    def detect_irrep(self, signature):
        return self.character_table.detect_irrep2(signature)

    def _detect_irreps(self):
        self.names_i = [
            self._detect_irrep(self.character_ig[i, :])
            for i in range(self.character_ig.shape[0])
        ]

    def class_id(self, classname):
        return self.names_g.index(classname)

    def _detect_irrep(self, signature):
        h = signature[self.class_id("E")]
        if h == 1:
            sig = "A"
        elif h == 2:
            sig = "E"
        elif h == 3:
            sig = "T"
        else:
            sig = "h" + str(h)

        return sig + str(signature)
        if np.all(signature == 1):
            return "A1"
        return str(h)
        # "AET"[len(signature)]
        h = signature[self.class_id("E")]

        # rotations = [ self.class_id(name) for name in self.names_g if ("C" in name) ]
        # rotations = self.class_id("6C2")
        ug = "g" if signature[self.class_id("i")] > 0 else "u"
        C = "?"
        N = ""
        if signature[self.class_id("6C4")] > 0:
            N = "1"
        else:
            N = "2"
        if h == 1:
            C = "A"
        if h == 2:
            C = "E"
            N = ""
        if h == 3:
            C = "T"

        return C + N + ug

    @property
    def ops_occ(self):  # XXX change to vv
        return self.spg_ops.O_svv

    def _build_character_table(self):
        # Diagonalize arbitrary Hamiltonian (the form 1/(1+g) is irrelevant)
        # to numerically build the character table. The degenerate eigenspaces
        # describe the irreducible representations.
        # There might be an accidental degeneracy, in which case the a representation
        # might end up being a direct product of two representations.
        H_oo = 1 / (self.g_o[self.inv_oo] + 1)
        groups_i = self.diagonalize_and_group(H_oo)
        self.groups_i = groups_i
        character_ig = np.zeros((len(groups_i), len(self.ops_g)), dtype=int)

        # For each group...
        for i, psi_no in enumerate(groups_i):
            # For each conjugacy class...
            for g, ops in enumerate(self.ops_g):
                # It is not necessary to loop over all group operations. However, it will be a good sanity check.
                # Calculate the trace of this irrep under operation o
                traces_o = np.array(
                    [
                        np.trace(
                            np.dot(psi_no[:, self.mul_oo[:, o]], psi_no.T.conjugate())
                        )
                        for o in ops
                    ]
                )
                h = len(psi_no) ** 0.5
                assert np.all(abs(traces_o - traces_o[0]) < 1e-10)
                character_ig[i, g] = int(np.round(traces_o[0] / h))
                # print(int(np.round(traces_o[0])) - traces_o[0])
                # assert(abs(int(np.round(traces_o[0])) - traces_o[0]) < 1e-10)

        self.character_ig = character_ig

    def diagonalize_and_group(self, H_oo):
        eps, psi = np.linalg.eigh(H_oo)
        groups = np.unique(abs(eps[:, None] - eps[None, :]) < 1e-6, axis=0)
        if self.verbose:
            print("Found %d irreducible representations." % len(groups))

        # Add eigenvales to lists
        groups_i = [[] for i in range(len(groups))]
        for irrep, eig_idx in zip(*np.where(groups)):
            groups_i[irrep].append(psi[:, eig_idx])

        return [np.array(x) for x in groups_i]


def analyze_symmetry(calc, verbose, layergroup=False):
    spg_ops = SPGOperations.from_atoms(
        calc.atoms, layergroup=layergroup, verbose=verbose
    )
    return
    pg = PointGroup(spg_ops, [0, 0, 1] if layergroup else None)
    results = []
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
                assert found is None
                found = irrep
    return results
