from dataclasses import dataclass
import numpy as np


def ppstr(W_cc, w_c=None):
    s = ""
    for i in range(3):
        s += "[ "
        for j in range(3):
            s += f"{W_cc[i, j]:5.2f} "
        s += "] "
        if w_c is None:
            s += "\n"
        else:
            s += f"  [ {w_c[i]:.15f} ]\n"
    return s + "\n"


@dataclass
class OperationInfo:
    op_cc: np.array
    cls: str
    axis: np.array

    def __post_init__(self):
        print('*** OperationInfo ***')
        print(self)
        print(self.op_cc)
        print()

    def is_clockwise(self, principal_axis):
        # Create arbitrary vector perpendicuylar to principal axis
        vec = np.linalg.qr(np.array([principal_axis]).T, mode='complete').Q[:,1]
        vec2 = self.op_cc @ vec
        return np.dot(np.cross(vec, vec2), principal_axis) < 0
        # Apply the vector
        # Calculate how much the vector rotated

    @classmethod
    def from_op(cls, op_cc):
        eigs_n, vecs_n = np.linalg.eig(op_cc)
        #print('in the beginning')
        #print(eigs_n, '\n', vecs_n)
        from functools import cmp_to_key
        eps = 1e-3

        def cmp(a, b):
            # First element is the index
            a, b = a[1], b[1]
            if abs(a.real - b.real) < eps:
                return -1 if a.imag < b.imag else (1 if a.imag > b.imag else 0)
            return -1 if a.real < b.real else 1

        idx = [i for i, _ in sorted(enumerate(eigs_n), key=cmp_to_key(cmp))]
        eigs_n = eigs_n[idx]
        vecs_n = vecs_n[:, idx]
        #print('after sort')
        #print(eigs_n,'\n', vecs_n)

        # Identity
        if np.allclose(eigs_n, 1.0):
            return cls(op_cc, "E", None)

        # Reflection
        if np.allclose(eigs_n, [-1.0, 1.0, 1.0]):
            print(vecs_n[0], 'axis of reflection')
            print('total op', op_cc)
            return cls(op_cc, "s", vecs_n[:, 0])

        # Inversion
        if np.allclose(eigs_n, [-1.0, -1.0, -1.0]):
            return cls(op_cc, "i", None)

        # Proper rotation
        for N in [2, 3, 4, 6]:
            angle = 2 * np.pi / N
            em, ep = np.exp(1j * angle), np.exp(-1j * angle)
            #print(eigs_n,'vs', [ ep, em, 1.0])
            if np.allclose(eigs_n, [ ep, em, 1.0]):
                return cls(op_cc, f"C{N}", vecs_n[:, 2])

        # Improper rotation
        for N in [2, 3, 4, 6]:
            angle = 2 * np.pi / N
            em, ep = np.exp(1j * angle), np.exp(-1j * angle)
            #print(eigs_n,'vs', [ -1, ep, em])
            if np.allclose(eigs_n, [ -1, ep, em]):
                return cls(op_cc, f"S{N}", vecs_n[:, 0])

        print(ppstr(op_cc))
        print(f"{eigs_n=} {vecs_n=}")
        raise NotImplementedError

    def __repr__(self):
        if self.cls == "E":
            return "Identity"
        if self.cls == "s":
            return f"Reflection over plane n=<{self.axis}>"
        if self.cls == "i":
            return "Inversion"
        if self.cls == 'C':
            return f"Rotation over axis n={self.axis}"
        return f"{self.cls} {self.axis}"

    @property
    def identity(self):
        return self.cls == "E"

    @property
    def rotation(self):
        if self.cls.startswith('C'):
            return True
        if self.cls.startswith('S'):
            return True
        return False

    @property
    def inversion(self):
        return self.cls == "i"

    @property
    def reflection(self):
        return self.cls == "s"

    @property
    def CN(self):
        # XXX Name, returns also order on S
        if self.cls.startswith('C'):
            return int(self.cls[1:])
        if self.cls.startswith('S'):
            return int(self.cls[1:])
        return None

    @property
    def N(self):
        if self.reflection:
            return 2
        return self.CN
