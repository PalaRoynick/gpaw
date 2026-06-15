# snippet-re-from-relaxed-start-student
label = ...  # Use the same label as in the previous exercise
# snippet-re-from-relaxed-end-student



label = "Si"
label = label  # Use the same label as in the previous exercise
# snippet-restart-from-relaxed-start

from ase.io import read
from gpaw import GPAW, PW, FermiDirac

# Add GPAW's relevant submodules again if you have restarted
# the kernel or if you are copy and pasting to a script

# read only the structure
atoms = read(label + '_gs.gpw')
# snippet-restart-from-relaxed-end




# snippet-lda-calculator-start
# self consistency in LDA
calc = GPAW(mode=PW(200),
            xc='LDA',
            kpts=(2, 2, 2),
            occupations=FermiDirac(0.01))
atoms.calc = calc
# snippet-lda-calculator-end

# snippet-homo-lumo-doc-start
# Run this cell to see the documentation of the get_homo_lumo method
...  # student: calc.get_homo_lumo?
# snippet-homo-lumo-doc-end


# sni-band-gap-start-student
# Potential energy
E = ...
vbm = ...
cbm = ...
# sni-band-gap-end-student

E = atoms.get_potential_energy()
vbm, cbm = calc.get_homo_lumo()

# snippet-band-gap-start
# Potential energy
E = E
vbm, cbm = vbm, cbm 

print('E=', E)
print('VBM=', vbm, 'CBM=', cbm)
print('band gap=', cbm - vbm)
# snippet-band-gap-end




# snippet-save-lda-start
# Save the ground state to file
calc.write(label + '_gs_LDA.gpw')
# snippet-save-lda-end


# sni-fixed-density-start-student
nbands = ... # Write the number of bands you are going to compute here, try 2x nbands
path = ...  # write your path here e.g. GXWKL/GMKG
convergence = ... # Your number of occupied orbitals comes here, e.g. 8/'occupied'
# sni-fixed-density-end-student


nbands = 16
path = 'GXWKL'
convergence = {'bands': 'occupied'}

# snippet-fixed-density-start
# Restart from ground state and fix potential:
calc = GPAW(label + '_gs_LDA.gpw').fixed_density(
    nbands=nbands,  
    symmetry='off',
    kpts={'path': path,  
          'npoints': 60},
    convergence=convergence
    )
# snippet-fixed-density-end



# snippet-band-structure-doc-start
# Have a look at the documentation of the band structure method
...  # student: calc.band_structure?
# snippet-band-structure-doc-end

# snippet-plot-band-structure-start
bs = calc.band_structure()
bs.plot(filename=label + '_bandstructure_LDA.png', show=True, emin = -10, emax=10)
# snippet-plot-band-structure-end

# snippet-save-band-structure-start
# You can also save the band structure data in a .json file
bs.write(label + '_bandstructure_LDA.json')
# snippet-save-band-structure-end
