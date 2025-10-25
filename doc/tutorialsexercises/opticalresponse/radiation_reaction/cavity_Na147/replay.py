import numpy as np
from ase.parallel import paropen
from gpaw.analyse.wignerseitz import wignerseitz
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Time propagation
td_calc = LCAOTDDFT('gs.gpw', parallel=parallel, txt='td.out')
td_calc.replay(name='wfs.ulm', update='all')
