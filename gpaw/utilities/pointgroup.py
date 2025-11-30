from dataclasses import dataclass
import numpy as np
from gpaw.utilities.pointgroup_operation import OperationInfo, ppstr


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
        return self.characters_ig * n_g[None, :] / self.order

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
        return self.normalized_characters_ig @ signature_g

    def detect_irrep2(self, signature_g):
        return np.linalg.solve(self.normalized_characters_ig2.T, signature_g)

    def print(self):
        try:
            print("%-20s" % "irreps/classes", end="")

            for name in self.classes:
                print("%-5s" % name, end="")
            print()
            for i, iname in enumerate(self.irreps):
                print("%-20s" % iname, end="")
                for g, gname in enumerate(self.classes):
                    print("%-5s" % ("%+02d" % self.characters_ig[i, g]), end="")
                print()
        except ValueError:
            print("...")


@dataclass
class SPGOperations:
    W_scc: np.array
    w_sc: np.array
    origin_shift_c: np.array
    cell_cv: np.array
    pointgroup: str
    allow_translations: bool = False

    def __post_init__(self):
        if not self.allow_translations:
            print(
                "TODO: Actually make sure that neglected translations are a normal subgroup of the operations"
            )
            print("Filtering out translations", len(self.w_sc))
            new_W_scc, new_w_sc = [], []
            for W_cc, w_c in zip(self.W_scc, self.w_sc):
                if np.allclose(w_c, 0, atol=0.01):
                    new_W_scc.append(W_cc)
                    new_w_sc.append(w_c)
            self.W_scc = new_W_scc
            self.w_sc = new_w_sc
            print("Left", len(self.w_sc))

            if not np.allclose(self.w_sc, 0):
                print(f"{self}")
                raise ValueError("We do not support translations atm.")

    @property
    def character_table(self):
        from gpaw.utilities.pointgroup_data import character_tables

        return CharacterTable.from_data(**character_tables[self.pointgroup])

    @classmethod
    def from_atoms(cls, atoms, verbose=False, *, layergroup: bool):
        # XXX: NOTE! This will modify the atoms!!!
        print("Modifying atoms with translations (for now, temporarily)")
        if layergroup:
            return cls.from_atoms_layergroup(atoms, verbose=verbose)
        else:
            return cls.from_atoms_spacegroup(atoms, verbose=verbose)

    @classmethod
    def from_atoms_spacegroup(cls, atoms, verbose=False):
        if verbose:
            print("Detecting spacegroup (NOT layergroup) with spglib...")
        import spglib

        dataset = spglib.get_symmetry_dataset(
            cell=(
                atoms.get_cell(),
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers(),
            ),
            symprec=1e-1,
        )
        print(f"{dataset=}")
        return cls.from_dataset(dataset, atoms, verbose=verbose)

    @classmethod
    def from_atoms_layergroup(cls, atoms, verbose=False):
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
            symprec=1e-1,
        )
        print(f"{dataset=}")
        return cls.from_dataset(dataset, atoms, verbose=verbose)

    @classmethod
    def from_dataset(cls, dataset, atoms, verbose=False):
        W_scc = dataset.rotations
        w_sc = dataset.translations
        origin_shift_c = dataset.origin_shift
        cell_cv = np.array(atoms.cell)
        # Kartik: Move to pointgroup_data
        dct = {
            "1": "C1",
            "-1": "Ci",
            "2": "C2",
            "m": "Cs",
            "2/m": "C2h",
            "222": "D2",
            "mm2": "C2v",
            "mmm": "D2h",
            "3": "C3",
            "-3": "S6",
            "32": "D3",
            "3m": "C3v",
            "-3m": "D3d",
            "4": "C4",
            "-4": "S4",
            "4/m": "C4h",
            "422": "D4",
            "4mm": "C4v",
            "-42m": "D2d",
            "4/mmm": "D4h",
            "6": "C6",
            "-6": "C3h",
            "6/m": "C6h",
            "622": "D6",
            "6mm": "C6v",
            "-6m2": "D3h",
            "6/mmm": "D6h",
            "23": "T",
            "m-3": "Th",
            "432": "O",
            "-43m": "Td",
            "m-3m": "Oh",
        }
        pointgroup = dct[dataset.pointgroup]
        if verbose:
            print(f"Pointgroup: {pointgroup} ({dataset.pointgroup})")
        unshifted = cls(
            W_scc, w_sc, origin_shift_c, cell_cv, pointgroup, allow_translations=True
        )
        print(f"unshifted {unshifted}")
        shifted = unshifted.apply_origin_shift(-origin_shift_c)
        # print('Atoms before', atoms.get_positions())
        # XXX: The apply origin shift has a negative sign
        # XXX: It is bad that we modify atoms here
        # XXX: All of this needs to be consolidated
        atoms.set_scaled_positions(atoms.get_scaled_positions() + origin_shift_c)

        # print('Atoms after', atoms.get_positions())
        # from ase.io import write
        # write('after.xyz', atoms)
        # asd
        print("shifts", shifted.w_sc)
        return shifted

    @property
    def O_svv(self):
        return np.array(
            [
                self.cell_cv.T @ W_cc @ np.linalg.inv(self.cell_cv).T
                for W_cc in self.W_scc
            ]
        )

    def apply_origin_shift(self, origin_shift_c):
        """
        spos'_c = W_cc spos_c + w_c

        spos'_c = W_cc (spos_c - origin_shift_c) + w_c + origin_shift_c
        """
        w_sc = (
            self.w_sc
            - np.einsum("scd,d->sc", self.W_scc, -origin_shift_c)
            + self.origin_shift_c
        )
        w_sc = (np.round(w_sc * 100) / 100) % 1.0 % 1.0
        return SPGOperations(
            self.W_scc,
            w_sc,
            self.origin_shift_c - origin_shift_c,
            self.cell_cv,
            self.pointgroup,
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
        principal_axis: list[float],
        verbose=True,
    ):
        self.operations = operations
        self.expected_classes = expected_classes
        self.principal_axis = principal_axis

        ops_occ = operations.ops_occ
        N = len(ops_occ)
        op_pool = range(N)
        self.ops_g = []

        self.g_o = np.zeros((N,), int)  # Conjugacy class index for each op
        while True:
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

        print(f"{operation_info_o=}")
        raise ValueError("Could not detect conjugacy class.")
        det_o = np.array([np.linalg.det(op_cc) for op_cc in ops_occ])
        eigs_o = np.array([np.sort(np.linalg.eig(op_cc)[0]) for op_cc in ops_occ])
        if len(det_o) == 1 and np.all(np.isclose(eigs_o, [-1, -1, 1])):
            return "1C2"
        if len(det_o) == 1 and np.all(np.isclose(eigs_o, [-1, -1, -1])):
            return "i"  # Inversion flips all axes, -x, -y, -z
        if len(det_o) == 1 and np.all(np.isclose(eigs_o, [1, 1, 1])):
            return "E"  # Identity leaves all axes intact x, y, z
        if len(det_o) == 3 and np.all(np.isclose(eigs_o, [-1, -1, 1])):
            return "3C2"  # C2 rotation along z is -x, -y, z
        if len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1, -1, 1])):
            return "6C2"
        if len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1, -1j, 1j])):
            return "6S4"  # -1j and 1j corresponds to 90 rotation. Determinant is -1 thus, improper.
        if np.all(np.isclose(eigs_o, [-1, 1, 1])):
            # Horizontal mirror operation flips one of the coordinates
            name = f"{len(det_o)}s"

            # if self.principal_axis is None:
            if len(det_o) in {6, 3}:
                return f"{len(det_o)}sd"
            # return name + 'h'

            reflection_type = ""
            for op_cc in ops_occ:
                eigs, vecs = np.linalg.eig(op_cc)
                index = np.argmin(eigs)

                # Reflection axis parallel to the reflection plane
                axis = vecs[:, index]
                print("principal axis", self.principal_axis)
                # print('op_cc', op_cc)
                print("op_cc", ppstr(op_cc))
                print("axis of reflection", axis)
                D = np.abs(np.dot(self.principal_axis, axis))
                if np.allclose(D, 1):
                    reflection_type += "h"
                elif np.allclose(D, 0):
                    reflection_type += "v"
                else:
                    raise ValueError("Unknown reflection type")
                print("D", D)
                print("reflection_type", reflection_type)
            if len(set(reflection_type)) == 1:
                name += reflection_type[0]
            else:
                name += reflection_type + "???"
            # self.used_class_names[name] += 1
            ## XXX Save something to self, which indicated the xz and yz planes
            # return name + [None, '_xz','_yz'][self.used_class_names[name]]
            return name

        if len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1, 1, 1])):
            return "6sd"
        if len(det_o) == 8 and np.all(
            np.isclose(
                eigs_o, [-1, np.exp(-1j * 2 * np.pi / 6), np.exp(1j * 2 * np.pi / 6)]
            )
        ):
            return "8S6"
        if len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1j, 1j, 1])):
            return "6C4"
        if len(det_o) == 8 and np.all(
            np.isclose(
                eigs_o, [np.exp(-1j * np.pi * 2 / 3), np.exp(1j * np.pi * 2 / 3), 1]
            )
        ):
            return "8C3"

        if np.all(np.isclose(det_o, -1)) and len(det_o) == 2:
            for N in [3, 4, 6]:
                c = np.cos(2 * np.pi / N)
                s = np.sin(2 * np.pi / N)
                eigs = sorted(
                    np.linalg.eig(np.array([[c, s, 0], [-s, c, 0], [0, 0, -1]]))[0]
                )
                if np.all(np.isclose(eigs, eigs_o)):
                    return f"2S{N}"
            print(f"{ops_occ=}")
            print(det_o, eigs_o)
            return "2S3"
        if np.all(np.isclose(det_o, 1)) and len(det_o) == 2:
            # XXX
            return "2C3"
        # if len(det_o) == 1 and np.all(np.isclose(eig_o, [-1, 1 ,1])):

        print(f"{ops_occ=}")
        print(f"{det_o=} {eigs_o=}")
        return "bug?"

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

        names_g = []
        for g, main_cc, operation_info_o, rotation_order in main_conjugacy_classes:
            info = operation_info_o[0]
            if info.rotation:
                # Rotation orthogonal to the main axis, add a prime
                if np.allclose(np.dot(info.axis, principal_axis), 0):
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

        # Find duplicate main classes to furher distinguish them
        from collections import Counter

        duplicates = [(k, v) for k, v in Counter(names_g).items() if v > 1]
        for name, count in duplicates:
            if name == "1sv" and count == 2:
                extras = ["_xz", "_yz"]
                # Actually fix according to molecular symmetry(?)
            elif name == "1C2" and count == 3:
                extras = ["_x", "_y", "_z"]
                # Actually fix according to crystal axes
            elif name == "1C2'" and count == 2:
                extras = ["_x", "_y"]
                # We should also name the unprimed 1C2 as _z now
            elif name == "1C3" and count == 2:
                extras = ["", "^2"]
                # TODO: Actually fix according to principal axis
            elif name == "1S3" and count == 2:
                extras = ["", "^5"]
                # TODO: Actually fix according to principal axis
            elif name == "1S4" and count == 2:
                extras = ["", "^3"]
            elif name == "1C4" and count == 2:
                extras = ["", "^3"]
                # TODO: Actually fix according to principal axis
            elif name == "2sv" and count == 2:
                extras = ["-2sv", "-2sd"]
                # TODO: Actually figure out which is sv and which is sd
            elif name == "2C2'" and count == 2:
                extras = ["", "'"]
                # TODO: Actually figure out which is ' and which is ''
            elif name == "1S6" and count == 2:
                extras = ["", "^5"]
                # TODO: Actually figure out which is S6 and which is S6^5
            elif name == "1C6" and count == 2:
                extras = ["", "^5"]
                # TODO: Actually figure out which is C6 and which is C6^5
            elif name == "3C2'" and count == 2:
                extras = ["", "'"]
                # TODO: Actually figure out which is 3C2' and which is 3C2''
            elif name == "3sv" and count == 2:
                extras = ["-3sv", "-3sd"]
                # TODO: Actually figure out which is 3sv and which is 3sd
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
    def __init__(self, spg_ops, principal_axis=[0, 0, 1]):
        self.spg_ops = spg_ops
        print("POINTGROUP", spg_ops.pointgroup)
        self.verbose = True
        self.principal_axis = principal_axis

        self.operations = SymmmetryOperations(self.ops_occ)

        # self._build_multiplication_table()
        character_table = self.spg_ops.character_table

        self.c4 = ConjugacyClassClassifierClass(
            self.operations,
            expected_classes=character_table.classes,
            principal_axis=principal_axis,
        )
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

    def signature(self, projectable):
        signature = np.zeros((len(self.names_g),), dtype=complex)
        for o, op_cc in enumerate(self.ops_occ):
            g = self.c4.g_o[o]
            signature[g] += (
                1
                / len(self.c4.ops_g[g])
                * projectable.dot(projectable.operation(op_cc))
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


class Projectable:
    def __init__(self, calc, cell_cv, wf, pseudo_wf=True):
        self.calc = calc
        self.cell_cv = cell_cv
        self.wf = wf
        self.pseudo_wf = pseudo_wf

    @classmethod
    def from_calc(cls, calc, n, pseudo_wf=True):
        if pseudo_wf:
            gamma = next(iter(calc.dft.ibzwfs))
            wf = gamma.psit_nX[n]  # get_pseudo_wave_function(n, grid_spacing=0.05)
        else:
            raise NotImplementedError()
            wf = calc.dft.ibzwfs.get_all_electron_wave_function(n, grid_spacing=0.05)
        return Projectable(calc, calc.atoms.cell, wf, pseudo_wf=pseudo_wf)

    def dot(self, projectable):
        result = self.wf.integrate(projectable.wf)
        return result

    def operation(self, op_vv):
        cell_cv = self.calc.atoms.cell

        # op_vv = self.cell_cv.T @ W_cc @ np.linalg.inv(self.cell_cv).T

        op_cc = np.linalg.inv(cell_cv.T) @ op_vv @ cell_cv.T
        op_cc = op_cc.T.copy()
        op_cc_int = np.asarray(np.round(op_cc), dtype=np.int64)
        assert np.allclose(op_cc, op_cc_int)

        if self.pseudo_wf:
            wf2 = self.wf.transform(op_cc_int)
            return Projectable(self.calc, cell_cv, wf2)
        else:
            # This one does not infact work
            raise NotImplementedError()
            wf = self.wf.copy()
            wf.symmetrize([op_cc_int], np.array([[0, 0, 0]], dtype=np.int64))
            return Projectable(self.calc, cell_cv, wf)


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
