import numpy as np

from gpaw.dft import DFT, Parameters
from gpaw.new.logger import Logger


def band_structure(dft: DFT,
                   bp,
                   *,
                   basis: str | dict[str | int | None, str] | None = None,
                   convergence: dict | None = None,
                   eigensolver: str | dict | None = None,
                   maxiter: int | None = None,
                   nbands: int | str | None = None,
                   parallel: dict | None = None,
                   random: bool | None = None,
                   txt='-'):
    old_params = dft.params.todict()
    old_params.pop('h', None)
    kwargs = {**old_params,
              'gpts': dft.density.nt_sR.desc.size,
              'kpts': bp,
              'symmetry': 'off'}
    _locals = locals()
    for key in ['basis',
                'convergence',
                'eigensolver',
                'maxiter',
                'nbands',
                'parallel',
                'random']:
        val = _locals[key]
        if val is not None:
            kwargs[key] = val
    params = Parameters(**kwargs)
    log = Logger(txt, dft.comm)
    builder = params.dft_component_builder(dft.atoms, log=log)
    basis_set = builder.create_basis_set()
    kbcomm1 = dft.ibzwfs.kpt_band_comm
    kbcomm2 = builder.communicators['D']
    potential = dft.potential.redist(
        builder.grid,
        builder.electrostatic_potential_desc,
        builder.atomdist,
        kbcomm1, kbcomm2)
    density = dft.density.redist(builder.grid,
                                 builder.interpolation_desc,
                                 builder.atomdist,
                                 kbcomm1, kbcomm2)
    ibzwfs = builder.create_ibz_wave_functions(basis_set, potential)
    for wfs in ibzwfs:
        wfs._occ_n = np.zeros(ibzwfs.nbands) + 1
        wfs.weight = 0.1
    ibzwfs.fermi_levels = dft.ibzwfs.fermi_levels
    scf_loop = builder.create_scf_loop()
    scf_loop.update_density_and_potential = False
    scf_loop.fix_fermi_level = True  # not update_fermi_level
    for name in ['energy', 'density', 'forces']:
        scf_loop.convergence.pop(name, None)

    scf_loop.hamiltonian.update_wave_functions(dft.ibzwfs)
    scf_loop.hamiltonian.update_wave_functions = lambda ibzwfs: None
    scf_loop.hamiltonian.nbzk = len(dft.ibzwfs.ibz.bz)
    dft = DFT.from_components(
        dft.atoms, ibzwfs, density, potential,
        builder.setups,
        scf_loop,
        builder.create_potential_calculator(),
        log,
        params=params,
        energies=dft.energies)

    return dft.ibzwfs
