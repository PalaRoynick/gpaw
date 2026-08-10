import taskblaster as tb
from ase.build import bulk


def materials():
    elements = ['Al', 'Si', 'Ti', 'Cu', 'Ag', 'Au', 'Pd', 'Pt']
    return {symbol: bulk(symbol) for symbol in elements}


workflow = tb.totree(materials(), name='material')
