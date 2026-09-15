from ase.data.g2_1 import atom_names, molecule_names
from myqueue.workflow import run

from gpaw.test.big.g2_1.g21 import relax


def workflow():
    deps = []
    for name in molecule_names + atom_names:
        d = run(function=relax, args=[name], name=name)
        deps.append(d)
    run(script='analyse.py', deps=deps)
