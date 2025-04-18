from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Time propagation
td_calc = LCAOTDDFT('gs.gpw', parallel=parallel, txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
WaveFunctionWriter(td_calc, 'wfs.ulm', interval=10)
td_calc.absorption_kick([0, 0, 1e-5])
td_calc.propagate(10, 3000)
td_calc.write('td.gpw', mode='all')
