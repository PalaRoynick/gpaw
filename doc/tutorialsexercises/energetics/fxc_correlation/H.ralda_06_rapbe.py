from gpaw.xc.fxc import FXCCorrelation

fxc = FXCCorrelation('H.ralda.pbe_wfcs.gpw', xc='rALDA',
                     txt='H.ralda_06_rapbe.output.txt',
                     ecut=300)
fxc.calculate()
