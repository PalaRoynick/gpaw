from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.qed import RRemission
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter

# Parallelzation settings
parallel = {'sl_auto': True, 'domain': 2, 'augment_grids': True}

# Time propagation
env = {'environment': 'waveguide',
       'quantization_plane': 35.05,
       'cavity_polarization': [0, 0, 1]}
td_calc = LCAOTDDFT('gs.gpw', parallel=parallel, txt='td.out',
                    rremission=RRemission(env))
DipoleMomentWriter(td_calc, 'dm.dat')
WaveFunctionWriter(td_calc, 'wfs.ulm', interval=10)
td_calc.absorption_kick([0, 0, 1e-5])
td_calc.propagate(10, 3000)
