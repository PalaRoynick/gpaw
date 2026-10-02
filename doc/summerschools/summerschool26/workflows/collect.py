from taskblaster.util import Collect

records = Collect('.').tree('*/relax')

for r in records:
    atoms = r.output
    print(atoms.symbols)
    print(atoms.cell.lengths())
    print()
