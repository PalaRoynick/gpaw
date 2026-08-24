import json
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from gpaw import KohnShamConvergenceError, setup_paths
from gpaw.atom.generator2 import generate
from gpaw.atom.tools import convergence_check
from gpaw.atom.tools import ecuts as ecuts_check
from gpaw.atom.tools import eos
from gpaw.dft import DFT, PW, MonkhorstPack, Parameters
from gpaw.mpi import broadcast, world
from gpaw.utilities.acwf import acwf_structures, check, reference_structure

PARAMS = {
    # 1-2:
    'H.1': '1s,s,p 0.9',
    'He.2': '1s,s,p 1.5',
    # 3-10:
    'Li.1': '2s,s,2p,D 3.0',
    'Li.3': '1s,2s,2p,p,D 1.5',
    'Be.2': '2s,s,2p 1.9',
    'Be.4': '1s,2s,2p,p,d 1.4',
    'B.3': '2s,s,2p,p,d 1.5',
    'C.4': '2s,s,2p,p,d 1.2',
    'N.5': '2s,5.6s,2p,5.6p,d 1.4,1.4,1.0 0.9',
    'O.6': '2s,s,2p,p,d,F 1.2',
    'F.7': '2s,s,2p,p,d 1.2,1.4',
    'Ne.8': '2s,s,2p,p,d 1.8',
    # 11-18:
    'Na.1': '3s,s,3p 2.6',
    'Na.9': '2s,3s,2p,3p,d,F 2.3',
    'Mg.2': '3s,s,3p,D 2.6',
    'Mg.10': '2s,3s,2p,3p,d,F 2.0,1.8',
    'Al.3': '3s,s,3p,p,d,F 2.1',
    'Si.4': '3s,s,3p,p,d,F 1.9',
    'P.5': '3s,s,3p,p,d,F 1.7',
    'S.6': '3s,s,3p,p,d,F 1.6',
    'Cl.7': '3s,s,3p,p,d,F 1.5',
    'Ar.8': '3s,s,3p,p,d,F 1.5',
    # 19-36:
    'K.1': '4s,s,4p,D 3.9',
    'K.9': '3s,4s,3p,4p,d,d,F 2.1',
    'Ca.2': '4s,s,4p 3.1',
    'Ca.10': '3s,4s,3p,4p,3d,d,F 2.1',
    'Sc.3': '4s,s,4p,p,3d,d,F 3.2',
    'Sc.11': '3s,4s,3p,4p,3d,d,F 2.3',
    'Ti.4': '4s,s,4p,p,3d,d 3.0',
    'Ti.12': '3s,4s,3p,4p,3d,d,F 2.2,2.2,2.3',
    'V.5': '4s,s,4p,p,3d,d 2.9',
    'V.13': '3s,4s,3p,4p,3d,d,F 2.1,2.1,2.3',
    'Cr.6': '4s,s,4p,p,3d,d 2.8',
    'Cr.14': '3s,4s,3p,4p,3d,d,F 2.1,2.1,2.3',
    'Mn.7': '4s,s,4p,p,3d,d 2.8',
    'Mn.15': '3s,4s,3p,4p,3d,d,F 2.0,2.0,2.2',
    'Fe.8': '4s,s,4p,p,3d,d 2.5',
    'Fe.16': '3s,4s,3p,4p,3d,d,F 2.1',
    'Co.9': '4s,s,4p,p,3d,d 2.2',
    'Co.17': '3s,4s,3p,4p,3d,d,F 2.1',
    'Ni.10': '4s,s,4p,p,3d,d 2.4',
    'Ni.18': '3s,4s,3p,4p,3d,d,F 2.0',
    'Cu.11': '4s,s,4p,p,3d,d 2.1',
    'Cu.19': '3s,4s,3p,4p,3d,d,F 1.9',
    'Zn.12': '4s,s,4p,p,3d 2.1',
    'Zn.20': '3s,4s,3p,4p,3d,d,F 1.9',
    'Ga.3': '4s,s,4p,p,d,F 2.2',
    'Ga.13': '4s,s,4p,p,3d,d,F 2.2',
    'Ge.4': '4s,s,4p,p,d,F 2.1',
    'Ge.14': '4s,s,4p,p,3d,d,F 2.1',
    'As.5': '4s,s,4p,p,d,F 2.0',
    'Se.6': '4s,s,4p,p,d,F 2.1',
    'Br.7': '4s,s,4p,p,d,F 2.1',
    'Kr.8': '4s,s,4p,p,d,F 2.1',
    # 37-54:
    'Rb.1': '5s,s,5p,D 3.9',
    'Rb.9': '4s,5s,4p,5p,d,d,F 2.5',
    'Sr.2': '5s,s,5p,D 3.9',
    'Sr.10': '4s,5s,4p,5p,4d,d,F 2.5',
    'Y.3': '5s,s,5p,p,4d,d,F 3.8',
    'Y.11': '4s,5s,4p,5p,4d,d,F 2.5',
    'Zr.4': '5s,s,5p,p,4d,d,F 3.5',
    'Zr.12': '4s,5s,4p,5p,4d,d,F 2.5',
    'Nb.5': '5s,s,5p,p,4d,d,F 3.4',
    'Nb.13': '4s,5s,4p,5p,4d,d,F 2.4,2.4,2.5',
    'Mo.6': '5s,s,5p,p,4d,d,F 3.4',
    'Mo.14': '4s,5s,4p,5p,4d,d,F 2.3',
    'Tc.7': '5s,s,5p,p,4d,d,F 3.3',
    'Tc.15': '4s,5s,4p,5p,4d,d,F 2.3',
    'Ru.8': '5s,s,5p,p,4d,d,F 3.3',
    'Ru.16': '4s,5s,4p,5p,4d,d,F 2.5',
    'Rh.9': '5s,s,5p,p,4d,d,F 3.3',
    'Rh.17': '4s,5s,4p,5p,4d,d,F 2.5',
    'Pd.10': '5s,s,5p,p,4d,d,F 3.3',
    'Pd.18': '4s,5s,4p,5p,4d,d,F 2.5',
    'Ag.11': '5s,s,5p,p,4d,d,F 3.3',
    'Ag.19': '4s,5s,4p,5p,4d,d,F 2.5',
    'Cd.12': '5s,s,5p,p,4d,d,F 3.3',
    'Cd.20': '4s,5s,4p,5p,4d,d,F 2.5',
    'In.13': '5s,s,5p,p,4d,d,F 2.6',
    'Sn.14': '5s,s,5p,p,4d,d,F 2.5',
    'Sb.15': '5s,s,5p,p,4d,d,F 2.5',
    'Te.6': '5s,6s,5p,p,d,d,F 2.5',
    'I.7': '5s,s,5p,p,d,F 2.4',
    'Xe.8': '5s,s,5p,p,d,F 2.3',
    # 55-56:
    'Cs.1': '6s,s,6p,5d,F 4.6',
    'Cs.9': '5s,6s,5p,6p,5d,0.5d,F 3.2',
    'Ba.2': '6s,1.0s,6p,1.0p,5d 4.9 4.1',
    'Ba.10': '5s,6s,5p,6p,5d,d,F 2.2',
    # 57-71:
    'La.11': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Ce.12': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Ce.12b': '5s,6s,5p,6p,5d,d,4f,f,G 2.6',
    'Pr.13': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Nd.14': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Pm.15': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Sm.16': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Eu.17': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Gd.18': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Tb.19': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Dy.20': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Ho.21': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Er.22': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Tm.23': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Yb.24': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    'Lu.25': '5s,6s,5p,6p,5d,d,4f,f,G 2.2',
    # 72-86:
    'Hf.4': '6s,s,6p,p,5d,d,F 3.3',
    'Hf.12': '5s,6s,5p,6p,5d,d,F 2.4',
    'Ta.5': '6s,s,6p,p,5d,d,F 3.3',
    'Ta.13': '5s,6s,5p,6p,5d,d,F 2.4',
    'W.6': '6s,s,6p,p,5d,d,F 3.1',
    'W.14': '5s,6s,5p,6p,5d,d,F 2.4',
    'Re.7': '6s,s,6p,p,5d,d,F 3.0',
    'Re.15': '5s,6s,5p,6p,5d,d,F 2.4',
    'Os.8': '6s,s,6p,p,5d,d,F 2.9',
    'Os.16': '5s,6s,5p,6p,5d,d,F 2.4',
    'Ir.9': '6s,s,6p,p,5d,d,F 2.9',
    'Ir.17': '5s,6s,5p,6p,5d,d,F 2.4',
    'Pt.10': '6s,s,6p,p,5d,d,F 2.9',
    'Pt.18': '5s,6s,5p,6p,5d,d,F 2.3',
    'Au.11': '6s,s,6p,p,5d,d,F 2.9',
    'Au.11b': '6s,s,6p,p,5d,d 2.9 2.7',
    'Au.19': '5s,6s,5p,6p,5d,d,F 2.3',
    'Hg.12': '6s,s,6p,p,5d,d 2.5',
    'Hg.20': '5s,6s,5p,6p,5d,d,F 2.3',
    'Tl.13': '6s,s,6p,p,5d,d,F 2.8',
    'Pb.14': '6s,s,6p,p,5d,d,F 2.6',
    'Bi.5': '6s,s,6p,p,d,F 2.8',
    'Bi.15': '6s,s,6p,p,5d,d,F 2.6',
    'Po.6': '6s,s,6p,p,d,F 2.7',
    'At.7': '6s,s,6p,p,d,F 2.6',
    'Rn.8': '6s,s,6p,p,d,F 2.6',
    # 87-88:
    'Fr.1': '7s,s,6p,5d,F 5.0',
    'Fr.9': '6s,7s,6p,7p,6d,d,F 2.7,2.5',
    'Ra.2': '6s,s,6p,5d 5.0',
    'Ra.10': '6s,7s,6p,7p,6d,d,F 2.7,2.5'}

PARAMS2 = {
    'H': '1s,s,p,D 1.3,1.2 1.1',
    'He': '1s,s,p,D 2.1,2.0 1.9',
    'Li': '1s,2s,2p,p,D 1.7,1.5 1.4',
    'Li.1e': '2s,s,2p,D 2.9,2.4 2.3',
    'Be.2e': '2s,s,2p,D 1.6,1.6 1.5',
    'Be': '1s,2s,2p,p,d,F 1.9,1.9,1.8 1.7',
    'B': '2s,s,2p,p,d,F 2.0,1.5,1.5 1.4',
    'C': '2s,3.0s,2p,3.0p,d,F 1.5,1.5,1.2 1.0',  # !!!
    'N': '2s,5.6s,2p,5.6p,d,F 1.4,1.4,1.0 0.9',
    'O': '2s,s,2p,p,d,F       1.4,1.4,1.4 0.9',
    'F': '2s,s,2p,p,d,F       1.2,1.4,1.4 1.1',
    'Ne': '2s,s,2p,p,d,F 2.7,2.8,2.0 1.9',
    'Na': '2s,3s,2p,3p,d,F 2.4,2.4,2.6 2.2',
    'Mg': '2s,3s,2p,3p,d,F 1.9,2.3,1.7 1.6',
    'Na.1e': '3s,s,3p,D 3.0,3.1 2.6',
    'Mg.2e': '3s,s,3p,D 2.4,2.5 2.2',
    'Al': '3s,s,3p,p,d,F 2.1,2.3,2.2 1.9',
    'Si': '3s,s,3p,p,d,F 1.9,1.9,1.9 1.7',
    'P': '3s,s,3p,p,d,F 2.3,2.1,1.7 1.6',
    'S': '3s,s,3p,p,d,F 2.3,1.8,1.6 1.5',
    'Cl': '3s,s,3p,p,d,F 2.1,2.0,1.5 1.4',
    'Ar': '3s,s,3p,p,d,F 1.9,1.8,1.8 1.7',
    'K': '3s,4s,3p,4p,d,d,F 2.1,2.1,2.1 1.9',
    'Ca': '3s,4s,3p,4p,3d,d,F 2.1,2.1,2.1 1.9',
    'Sc': '3s,4s,3p,4p,3d,d,F 2.3,2.3,2.3 2.1',
    'Ti': '3s,4s,3p,4p,3d,d,F 2.0,2.0,2.1 1.8',
    'V': '3s,4s,3p,4p,3d,d,F 1.9,2.0,2.1 1.7',
    'Cr': '3s,4s,3p,4p,3d,d,F 2.0,2.3,2.1 1.8',
    'Mn': '3s,4s,3p,4p,3d,d,F 1.9,1.8,2.0 1.7',
    'Fe': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.0 1.8',
    'Co': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.0 1.8',
    'Ni': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.1 1.8',
    'Cu': '3s,4s,3p,4p,3d,d,F 1.9,1.9,2.0 1.7',
    'Zn': '3s,4s,3p,4p,3d,d,F 1.9,1.9,2.1 1.7',
    'Sc.3e': '4s,s,4p,p,3d,d,F 2.8,2.8,2.8 2.7',
    'Ti.4e': '4s,s,4p,p,3d,d,F 2.7,2.7,2.7 2.6',
    'V.5e': '4s,s,4p,p,3d,d,F 2.8,2.8,2.8 2.7',  # !!!
    'Cr.6e': '4s,s,4p,p,3d,d,F 2.5,2.7,2.7 2.4',
    'Mn.7e': '4s,s,4p,p,3d,d,F 2.4,2.4,2.3 2.5',  # !!!
    'Fe.8e': '4s,s,4p,p,3d,d,F 2.6,3.2,2.4 2.3',
    'Co.9e': '4s,s,4p,p,3d,d,F 2.1,2.1,2.1 2.0',
    'Ni.10e': '4s,s,4p,p,3d,d,F 2.2,2.2,2.2 2.0',
    'Cu.11e': '4s,s,4p,p,3d,d,F 2.2,2.1,2.1 2.3',  # !!!
    'Zn.12e': '4s,s,4p,p,3d,F 2.0,2.0,2.0 1.9',
    'Ga': '4s,s,4p,p,3d,d,F 2.1,2.1,2.1 1.9',
    'Ge': '4s,s,4p,p,3d,d,F 2.2,2.2,2.1 1.9',
    'Ga.3e': '4s,s,4p,p,d,F 2.3,2.3,2.2 2.0',
    'Ge.4e': '4s,s,4p,p,d,F 2.1,2.1,2.1 1.9',
    'As': '4s,s,4p,p,d,F 2.0,2.0,2.0 1.8',
    'Se': '4s,s,4p,p,d,F 2.1,2.1,2.4 1.9',
    'Br': '4s,s,4p,p,d,F 2.1,2.1,2.1 1.9',
    'Kr': '4s,s,4p,p,d,F 2.1,2.1,2.1 1.9',
    'Rb': '4s,5s,4p,5p,d,d,F 2.7,2.6,2.5 2.2',  # !!!
    'Sr': '4s,5s,4p,5p,4d,d,F 3.1,2.8,2.9 2.5',
    'Y': '4s,5s,4p,5p,4d,d,F 2.5,2.5,2.3 2.2',
    'Zr': '4s,5s,4p,5p,4d,d,F 2.8,2.3,2.4 2.1',
    'Nb': '4s,5s,4p,5p,4d,d,F 2.2,2.5,2.3 2.1',
    'Mo': '4s,5s,4p,5p,4d,d,F 2.3,2.3,2.3 2.1',
    'Tc': '4s,5s,4p,5p,4d,d,F 2.4,2.4,2.4 2.2',
    'Ru': '4s,5s,4p,5p,4d,d,F 2.3,2.6,2.4 2.1',
    'Rh': '4s,5s,4p,5p,4d,d,F 2.3,2.4,2.4 2.1',
    'Pd': '4s,5s,4p,5p,4d,d,F 2.5,2.5,2.5 2.2',
    'Ag': '4s,5s,4p,5p,4d,d,F 2.4,2.5,2.5 2.2',
    'Pd.10e': '5s,s,5p,p,4d,d,F 3.1,3.2,3.1 2.9',  # !!!
    'Ag.11e': '5s,s,5p,p,4d,d,F 3.9,3.7,3.6 3.0',  # !!!
    'Cd.12e': '5s,s,5p,p,4d,d,F 2.6,2.4,2.6 2.3',
    'In': '5s,s,5p,p,4d,d,F 2.6,2.6,2.6 2.3',
    'Sn': '5s,s,5p,p,4d,d,F 2.5,2.5,2.5 2.2',
    'Sb': '5s,s,5p,p,4d,d,F 2.6,2.4,3.2 2.3',
    'Te': '5s,6s,5p,p,d,d,F 2.7,2.5,3.5 2.1',
    'I': '5s,s,5p,p,d,F 2.4,2.3,2.6 2.2',
    'Xe': '5s,s,5p,p,d,F 2.2,2.3,2.2 2.1',
    'Cs': '5s,6s,5p,6p,5d,0.5d,F 3.2,3.2,3.2 2.9',  # !!!
    'Ba': '5s,6s,5p,6p,5d,d,F 1.8,1.9,2.0 1.7',  # !!!
    'La': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Ce': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Pr': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',  # !!!
    'Nd': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Pm': '5s,6s,5p,6p,5d,d,4f,f,G 3.4,2.2,2.0,2.1 1.9',
    'Sm': '5s,6s,5p,6p,5d,d,4f,f,G 3.5,2.0,1.9,2.2 1.8',
    'Eu': '5s,6s,5p,6p,5d,d,4f,f,G 3.9,2.1,2.1,2.3 2.0',
    'Gd': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Tb': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Dy': '5s,6s,5p,6p,5d,d,4f,f,G 3.8,2.0,2.1,2.1 1.9',
    'Ho': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Er': '5s,6s,5p,6p,5d,d,4f,f,G 2.8,2.4,2.2,2.7 2.1',
    'Tm': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Yb': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Lu': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Hf': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Ta': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'W': '   5s,6s,5p,6p,5d,d,F 2.3,2.4,2.5 2.2',
    'Re': '  5s,6s,5p,6p,5d,d,F 2.5,2.4,2.5 2.2',
    'Os': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Ir': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Pt': '  5s,6s,5p,6p,5d,d,F 2.4,2.5,2.4 2.2',
    'Au': '  5s,6s,5p,6p,5d,d,F 2.4,2.6,2.5 2.2',
    'Hg': '  5s,6s,5p,6p,5d,d,F 2.3,2.3,2.3 2.1',
    'Os.8e': ' 6s,s,6p,p,5d,d,F 2.9,2.9,2.9 2.6',
    'Ir.9e': ' 6s,s,6p,p,5d,d,F 2.9,2.9,2.9 2.6',
    'Pt.10e': '6s,s,6p,p,5d,d,F 2.9,2.9,2.9 2.6',
    'Au.11e': '6s,s,6p,p,5d,d,F 2.7,2.7,2.7 2.5',
    'Hg.12e': '6s,s,6p,p,5d,d,F 2.6,2.5,2.8 2.3',
    'Tl': '6s,s,6p,p,5d,d,F 2.9,2.5,2.7 2.4',
    'Bi': '6s,s,6p,p,5d,d,F 2.7,2.8,3.0 2.4',
    'Bi.5e': '6s,s,6p,p,d,F 2.8,2.8,2.8 2.5',
    'Po': '6s,s,6p,p,d,F 2.5,2.5,4.2 2.3',
    'At': '6s,s,6p,p,d,F 2.5,2.6,3.5 2.1',
    'Rn': '6s,s,6p,p,d,F 2.3,2.8,2.4 2.1',
    'Fr': '6s,7s,6p,7p,6d,d,F 2.5,2.4,2.8 2.1',
    'Ra': '6s,7s,6p,7p,6d,d,F 2.9,2.4,2.3 2.1'}

PARAMS8 = {
    'H': '1s,s,p,D 1.2,1.1 1.0',
    'H.a': '1s,s,p,D 1.1,1.1 1.0',
    'He': '1s,s,p,D 1.7,1.5 1.4',
    'He.a': '1s,s,p,D 1.6,1.4 1.3',
    'Li': '1s,2s,2p,p,D 1.7,1.5 1.4',
    'Be': '1s,2s,2p,p,d,F 1.9,1.9,1.8 1.7',
    'B': '2s,s,2p,p,d,F 2.0,1.5,1.5 1.4',
    'C': '2s,3.0s,2p,3.0p,d,F 1.5,1.5,1.2 1.0',  # !!!
    'N': '2s,5.6s,2p,5.6p,d,F 1.4,1.4,1.0 0.9',
    'O': '2s,s,2p,p,d,F       1.4,1.4,1.4 0.9',
    'O.a': '2s,5.6s,2p,5.6p,d,F 1.4,1.4,1.4 0.9',
    'O.b': '2s,s,2p,p,d,F       1.4,1.4,1.0 0.9',
    'F': '2s,s,2p,p,d,F       1.2,1.4,1.4 1.1',
    'F.a': '2s,5.6s,2p,5.6p,d,F       1.2,1.4,1.4 1.1',
    'F.b': '2s,s,2p,p,d,F       1.2,1.4,1.3 1.0',
    'Ne': '2s,s,2p,p,d,F 2.4,2.4,2.0 1.9',
    'Ne.a': '2s,s,2p,p,d,F 2.3,2.3,2.0 1.8',
    'Na': '2s,3s,2p,3p,d,F 2.4,2.4,2.6 2.2',
    'Na.a': '2s,3s,2p,3p,d,F 2.3,2.3,2.5 2.1',
    'Mg': '2s,3s,2p,3p,d,F 2.4,2.4,2.6 2.2',
    'Mg.a': '2s,3s,2p,3p,d,F 2.3,2.3,2.5 2.1',
    'Al': '3s,s,3p,p,d,F 2.1,2.3,2.2 1.9',
    'Si': '3s,s,3p,p,d,F 1.9,1.9,1.9 1.7',
    'P': '3s,s,3p,p,d,F 2.3,2.1,1.7 1.6',
    'S': '3s,s,3p,p,d,F 2.3,1.8,1.6 1.5',
    'Cl': '3s,s,3p,p,d,F 2.1,2.0,1.5 1.4',
    'Cl.a': '3s,s,3p,p,d,F 2.0,1.9,1.5 1.4',
    'Ar': '3s,s,3p,p,d,F 1.9,1.8,1.8 1.6',
    'K': '3s,4s,3p,4p,d,d,F 2.0,2.0,2.0 1.8',
    'Ca': '3s,4s,3p,4p,3d,d,F 2.1,2.1,2.1 1.9',
    'Sc': '3s,4s,3p,4p,3d,d,F 2.3,2.3,2.3 2.1',
    'Ti': '3s,4s,3p,4p,3d,d,F 2.0,2.0,2.1 1.8',
    'V': '3s,4s,3p,4p,3d,d,F 1.9,2.0,2.1 1.7',
    'Cr': '3s,4s,3p,4p,3d,d,F 2.0,2.3,2.1 1.8',
    'Cr.a': '3s,4s,3p,4p,3d,d,F 1.9,1.9,2.1 1.7',
    'Mn': '3s,4s,3p,4p,3d,d,F 1.9,1.8,2.0 1.7',
    'Mn.a': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.0 1.8',
    'Fe': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.0 1.8',
    'Co': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.0 1.8',
    'Ni': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.1 1.8',
    'Cu': '3s,4s,3p,4p,3d,d,F 1.9,1.9,2.0 1.7',
    'Zn': '3s,4s,3p,4p,3d,d,F 1.9,1.9,2.1 1.7',
    'Ga': '4s,s,4p,p,3d,d,F 2.1,2.1,2.1 1.9',
    'Ge': '4s,s,4p,p,3d,d,F 2.2,2.2,2.1 1.9',
    'As': '4s,s,4p,p,d,F 2.0,2.0,2.0 1.8',
    'Se': '4s,s,4p,p,d,F 2.1,2.1,2.4 1.9',
    'Br': '4s,s,4p,p,d,F 2.0,2.0,2.1 1.8',
    'Kr': '4s,s,4p,p,d,F 2.0,2.0,2.1 1.8',
    'Rb': '4s,5s,4p,5p,d,d,F 2.5,2.3,2.3 2.1',  # !!!
    'Rb.a': '4s,5s,4p,5p,d,d,F 2.4,2.2,2.2 2.0',  # !!!
    'Sr': '4s,5s,4p,5p,4d,d,F 2.5,2.3,2.3 2.1',
    'Y': '4s,5s,4p,5p,4d,d,F 2.5,2.3,2.3 2.1',
    'Zr': '4s,5s,4p,5p,4d,d,F 2.8,2.3,2.4 2.1',
    'Nb': '4s,5s,4p,5p,4d,d,F 2.2,2.5,2.3 2.1',
    'Mo': '4s,5s,4p,5p,4d,d,F 2.3,2.3,2.3 2.1',
    'Tc': '4s,5s,4p,5p,4d,d,F 2.4,2.4,2.4 2.2',
    'Ru': '4s,5s,4p,5p,4d,d,F 2.3,2.6,2.4 2.1',
    'Rh': '4s,5s,4p,5p,4d,d,F 2.3,2.4,2.4 2.1',
    'Pd': '4s,5s,4p,5p,4d,d,F 2.5,2.5,2.5 2.2',
    'Ag': '4s,5s,4p,5p,4d,d,F 2.4,2.5,2.5 2.2',
    'Cd': '5s,s,5p,p,4d,d,F 2.6,2.4,2.6 2.3',
    'In': '5s,s,5p,p,4d,d,F 2.6,2.6,2.6 2.3',
    'Sn': '5s,s,5p,p,4d,d,F 2.5,2.5,2.5 2.2',
    'Sb': '5s,s,5p,p,4d,d,F 2.6,2.4,3.2 2.3',
    'Te': '5s,6s,5p,p,d,d,F 2.7,2.5,3.5 2.1',
    'I': '5s,s,5p,p,d,F 2.4,2.3,2.6 2.2',
    'Xe': '5s,s,5p,p,d,F 2.2,2.3,2.2 2.1',
    'Cs': '5s,6s,5p,6p,5d,d,F 2.1,2.1,2.1 2.0',  # !!!
    'Cs.a': '5s,6s,5p,6p,5d,d,f,G 2.1,2.1,2.1 2.0',  # !!!
    'Ba': '5s,6s,5p,6p,5d,d,f,G 2.1,2.1,2.1 2.0',  # OK
    'La': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Ce': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Pr': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Nd': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Pm': '5s,6s,5p,6p,5d,d,4f,f,G       3.4,2.3,2.0,2.1 1.8',
    'Sm': '5s,6s,5p,6p,5d,d,4f,f,G       3.5,2.0,2.1,2.2 1.7',
    'Eu': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Gd': '5s,6s,5p,6p,5d,d,4f,f,G       2.2,2.2,2.2,2.2 2.0',
    'Gd.a': '5s,6s,5p,6p,5d,d,4f,f,G     3.0,2.6,2.5,2.4 2.1',
    'Tb': '5s,6s,5p,6p,5d,d,4f,f,G       2.2,2.2,2.2,2.2 2.0',
    'Dy': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Dy.a': '5s,6s,5p,6p,5d,d,4f,f,G     3.0,2.6,2.5,2.4 2.0',
    'Ho': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Er': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Tm': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Yb': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Lu': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Hf': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Ta': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'W': '   5s,6s,5p,6p,5d,d,F 2.3,2.4,2.5 2.2',
    'Re': '  5s,6s,5p,6p,5d,d,F 2.5,2.4,2.5 2.2',
    'Os': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Ir': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Pt': '  5s,6s,5p,6p,5d,d,F 2.4,2.5,2.4 2.2',
    'Au': '  5s,6s,5p,6p,5d,d,F 2.4,2.6,2.5 2.2',
    'Hg': '5s,6s,5p,6p,5d,d,F 2.4,2.6,2.5 2.2',
    'Tl': '6s,s,6p,p,5d,d,F 2.9,2.5,2.7 2.4',
    'Bi': '6s,s,6p,p,5d,d,F 2.7,2.8,3.0 2.4',
    'Po': '6s,s,6p,p,d,F 2.5,2.5,4.2 2.3',
    'At': '6s,s,6p,p,d,F 2.5,2.6,3.5 2.1',
    'Rn': '6s,s,6p,p,d,F 2.3,2.8,2.4 2.1',
    'Fr': '6s,7s,6p,7p,6d,d,F 2.5,2.4,2.8 2.1',
    'Ra': '6s,7s,6p,7p,6d,d,F 2.5,2.4,2.3 2.1',
    'Ra.a': '6s,7s,6p,7p,6d,d,F 2.4,2.3,2.3 2.1'}

PARAMS9 = {
    'H': '1s,s,p,D 1.2,1.1 1.0',
    'He': '1s,s,p,D 1.7,1.5 1.4',
    'Li': '1s,2s,2p,p,D 1.7,1.5 1.4',
    'Be': '1s,2s,2p,p,d,F 1.9,1.9,1.8 1.7',
    'B': '2s,s,2p,p,d,F 2.0,1.5,1.5 1.4',
    'C': '2s,3.0s,2p,3.0p,d,F 1.4,1.4,1.2 1.0',  # !!!
    'N': '2s,5.6s,2p,5.6p,d,F 1.4,1.4,1.0 0.9',
    'O': '2s,5.6s,2p,5.6p,d,F 1.4,1.4,1.1 0.9',
    'F': '2s,5.6s,2p,5.6p,d,F       1.2,1.4,1.4 1.1',
    'Ne': '2s,s,2p,p,d,F 2.4,2.4,2.0 1.9',
    'Na': '2s,3s,2p,3p,d,F 2.4,2.4,2.6 2.2',
    'Mg': '2s,3s,2p,3p,d,F 2.4,2.4,2.6 2.2',
    'Al': '3s,s,3p,p,d,F 2.1,2.3,2.2 1.9',
    'Si': '3s,s,3p,p,d,F 1.9,1.9,1.9 1.7',
    'P': '3s,s,3p,p,d,F 2.3,2.1,1.7 1.6',
    'S': '3s,s,3p,p,d,F 2.3,1.8,1.6 1.5',
    'Cl': '3s,s,3p,p,d,F 2.1,2.0,1.5 1.4',
    'Ar': '3s,s,3p,p,d,F 1.9,1.8,1.8 1.6',
    'K': '3s,4s,3p,4p,d,d,F 2.0,2.0,2.0 1.8',
    'Ca': '3s,4s,3p,4p,3d,d,F 2.1,2.1,2.1 1.9',
    'Sc': '3s,4s,3p,4p,3d,d,F 2.3,2.3,2.3 2.1',
    'Ti': '3s,4s,3p,4p,3d,d,F 2.0,2.0,2.1 1.8',
    'V': '3s,4s,3p,4p,3d,d,F 1.9,2.0,2.1 1.7',
    'Cr': '3s,4s,3p,4p,3d,d,F 2.0,2.3,2.1 1.8',
    'Mn': '3s,4s,3p,4p,3d,d,F 1.9,1.8,2.0 1.7',
    'Fe': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.0 1.8',
    'Co': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.0 1.8',
    'Ni': '3s,4s,3p,4p,3d,d,F 2.0,1.9,2.1 1.8',
    'Cu': '3s,4s,3p,4p,3d,d,F 1.9,1.9,2.0 1.7',
    'Zn': '3s,4s,3p,4p,3d,d,F 1.9,1.9,2.1 1.7',
    'Ga': '4s,s,4p,p,3d,d,F 2.1,2.1,2.1 1.9',
    'Ge': '4s,s,4p,p,3d,d,F 2.2,2.2,2.1 1.9',
    'As': '4s,s,4p,p,d,F 2.0,2.0,2.0 1.8',
    'Se': '4s,s,4p,p,d,F 2.1,2.1,2.4 1.9',
    'Br': '4s,s,4p,p,d,F 2.0,2.0,2.1 1.8',
    'Kr': '4s,s,4p,p,d,F 2.0,2.0,2.1 1.8',
    'Rb': '4s,5s,4p,5p,d,d,F 2.5,2.3,2.3 2.1',  # !!!
    'Sr': '4s,5s,4p,5p,4d,d,F 2.5,2.3,2.3 2.1',
    'Y': '4s,5s,4p,5p,4d,d,F 2.5,2.3,2.3 2.1',
    'Zr': '4s,5s,4p,5p,4d,d,F 2.8,2.3,2.4 2.1',
    'Nb': '4s,5s,4p,5p,4d,d,F 2.2,2.5,2.3 2.1',
    'Mo': '4s,5s,4p,5p,4d,d,F 2.3,2.3,2.3 2.1',
    'Tc': '4s,5s,4p,5p,4d,d,F 2.4,2.4,2.4 2.2',
    'Ru': '4s,5s,4p,5p,4d,d,F 2.3,2.6,2.4 2.1',
    'Rh': '4s,5s,4p,5p,4d,d,F 2.3,2.4,2.4 2.1',
    'Pd': '4s,5s,4p,5p,4d,d,F 2.5,2.5,2.5 2.2',
    'Ag': '4s,5s,4p,5p,4d,d,F 2.4,2.5,2.5 2.2',
    'Cd': '5s,s,5p,p,4d,d,F 2.6,2.4,2.6 2.3',
    'In': '5s,s,5p,p,4d,d,F 2.6,2.6,2.6 2.3',
    'Sn': '5s,s,5p,p,4d,d,F 2.5,2.5,2.5 2.2',
    'Sb': '5s,s,5p,p,4d,d,F 2.6,2.4,3.2 2.3',
    'Te': '5s,6s,5p,p,d,d,F 2.7,2.5,3.5 2.1',
    'I': '5s,s,5p,p,d,F 2.4,2.3,2.6 2.2',
    'Xe': '5s,s,5p,p,d,F 2.2,2.3,2.2 2.1',
    'Cs': '5s,6s,5p,6p,5d,d,F 2.1,2.1,2.1 2.0',  # !!!
    'Ba': '5s,6s,5p,6p,5d,d,f,G 2.1,2.1,2.1 2.0',  # OK
    'La': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Ce': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Pr': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Nd': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Pm': '5s,6s,5p,6p,5d,d,4f,f,G       3.4,2.3,2.0,2.1 1.8',
    'Sm': '5s,6s,5p,6p,5d,d,4f,f,G       3.5,2.0,2.1,2.2 1.7',
    'Eu': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Gd': '5s,6s,5p,6p,5d,d,4f,f,G       2.2,2.2,2.2,2.2 2.0',
    'Tb': '5s,6s,5p,6p,5d,d,4f,f,G       2.2,2.2,2.2,2.2 2.0',
    'Dy': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Ho': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Er': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Tm': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Yb': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Lu': '5s,6s,5p,6p,5d,d,4f,f,G       2.3,2.2,2.1,2.1 2.0',
    'Hf': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Ta': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'W': '   5s,6s,5p,6p,5d,d,F 2.3,2.4,2.5 2.2',
    'Re': '  5s,6s,5p,6p,5d,d,F 2.5,2.4,2.5 2.2',
    'Os': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Ir': '  5s,6s,5p,6p,5d,d,F 2.4,2.4,2.4 2.2',
    'Pt': '  5s,6s,5p,6p,5d,d,F 2.4,2.5,2.4 2.2',
    'Au': '  5s,6s,5p,6p,5d,d,F 2.4,2.6,2.5 2.2',
    'Hg': '5s,6s,5p,6p,5d,d,F 2.4,2.6,2.5 2.2',
    'Tl': '6s,s,6p,p,5d,d,F 2.9,2.5,2.7 2.4',
    'Bi': '6s,s,6p,p,5d,d,F 2.7,2.8,3.0 2.4',
    'Po': '6s,s,6p,p,d,F 2.5,2.5,4.2 2.3',
    'At': '6s,s,6p,p,d,F 2.5,2.6,3.5 2.1',
    'Rn': '6s,s,6p,p,d,F 2.3,2.8,2.4 2.1',
    'Fr': '6s,7s,6p,7p,6d,d,F 2.5,2.4,2.8 2.1',
    'Ra': '6s,7s,6p,7p,6d,d,F 2.5,2.4,2.3 2.1'}

EXTRA = {
    'H.b': '1s,s,p,D 1.2,1.1 1.0',
    'He.b': '1s,s,p,D 2.0,1.9 1.8',
    'N.b': '2s,3.0s,2p,3.0p,d,F 1.4,1.4,1.0 0.9',
    'N.c': '2s,3.0s,2p,3.0p,d,F 1.4,1.4,1.0 0.8',
    'O.b': '2s,3.0s,2p,3.0p,d,F 1.4,1.4,1.4 0.9',
    'O.c': '2s,5.0s,2p,5.0p,d,F 1.4,1.4,1.4 0.9',
    'O.d': '2s,s,2p,p,d,F 1.4,1.4,1.4 1.0',
    'Cl.b': '3s,s,3p,p,d,F 1.9,1.8,1.5 1.4',
    'As.b': '4s,s,4p,p,d,F 2.0,2.0,2.0 1.9',
    'Cs.b': '5s,6s,5p,6p,5d,d,F 3.2,3.2,3.2 2.9',  # !!!
    'Cs.c': '5s,6s,5p,6p,5d,0.5d,F 2.4,2.4,2.4 2.3',  # !!!
    'Cs.d': '5s,6s,5p,6p,5d,0.5d,F 2.5,2.5,2.5 2.3',  # !!!
    'Ba.b': '5s,6s,5p,6p,5d,d,F 2.1,2.1,2.1 2.0',  # !!!
    'Ba.c': '5s,6s,5p,6p,5d,d,f,G 2.1,2.1,2.1 2.0',  # OK
    'Ba.d': '5s,6s,5p,6p,5d,d,f,G 2.5,2.5,2.5 2.3',  # !!!
    'La.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',  # OK
    'Ce.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Pr.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',  # !!!
    'Nd.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Pm.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Sm.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Eu.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Gd.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Tb.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Dy.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Ho.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Er.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Tm.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Yb.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Lu.b': '5s,6s,5p,6p,5d,d,4f,f,G 3.0,2.2,2.2,2.2 2.0',
    'Nb.b': '4s,5s,4p,5p,4d,d,F 2.3,2.3,2.3 2.0',
    'Tc.b': '4s,5s,4p,5p,4d,d,F 2.3,2.3,2.3 2.0',
    'Rh.b': '4s,5s,4p,5p,4d,d,F 2.3,2.3,2.3 2.0',
    'Au.b': '5s,6s,5p,6p,5d,d,F 2.4,2.5,2.4 2.2',
    'Hg.b': '5s,6s,5p,6p,5d,d,F 2.4,2.5,2.4 2.2'}

PARAMS5 = {
    'H': '1s,s,p,D 1.2,1.1 1.0',
    'He': '1s,s,p,D 1.8,1.8 2.0',
    'N': '2s,3.0s,2p,3.0p,d,F 1.4,1.4,1.0 0.9',
    'O': '2s,s,2p,p,d,F 1.4,1.4,1.4 1.0',
    'F': '2s,s,2p,p,d,F       1.2,1.4,1.4 1.1',
    'Cl': '3s,s,3p,p,d,F 1.7,1.7,1.7 1.5',
    'As': '4s,s,4p,p,d,F 2.0,2.0,2.0 1.9',
    'Cs': '5s,6s,5p,6p,5d,0.5d,F 2.3,2.3,2.3 2.2',
    'Ba': '5s,6s,5p,6p,5d,d,f,G 2.0,2.0,2.0 1.9',
    'La': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Ce': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Pr': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Nd': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Pm': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Sm': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Eu': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Gd': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Tb': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Dy': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Ho': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Er': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Tm': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Yb': '5s,6s,5p,6p,5d,d,4f,f,G 2.1,2.1,2.1,2.1 2.0',
    'Lu': '5s,6s,5p,6p,5d,d,4f,f,G 2.2,2.2,2.2,2.2 2.0',
    'Hg': '5s,6s,5p,6p,5d,d,F 2.2,2.2,2.2 2.0'}

PARAMS5SOFT = {
    'Li.1e': '2s,s,2p,D 2.9,2.4 2.3',
    'Be.2e': '2s,s,2p,D 1.6,1.6 1.5',
    'Na.1e': '3s,s,3p,D 3.0,3.1 2.6',
    'Mg.2e': '3s,s,3p,D 2.4,2.5 2.2',
    'Sc.3e': '4s,s,4p,p,3d,d,F 2.8,2.8,2.8 2.7',
    'Ti.4e': '4s,s,4p,p,3d,d,F 2.7,2.7,2.7 2.6',
    'V.5e': '4s,s,4p,p,3d,d,F 2.8,2.8,2.8 2.7',  # !!!
    'Cr.6e': '4s,s,4p,p,3d,d,F 2.5,2.7,2.7 2.4',
    'Mn.7e': '4s,s,4p,p,3d,d,F 2.4,2.4,2.3 2.5',  # !!!
    'Fe.8e': '4s,s,4p,p,3d,d,F 2.6,3.2,2.4 2.3',
    'Co.9e': '4s,s,4p,p,3d,d,F 2.1,2.1,2.1 2.0',
    'Ni.10e': '4s,s,4p,p,3d,d,F 2.2,2.2,2.2 2.0',
    'Cu.11e': '4s,s,4p,p,3d,d,F 2.2,2.1,2.1 2.3',  # !!!
    'Zn.12e': '4s,s,4p,p,3d,F 2.0,2.0,2.0 1.9',
    'Ga.3e': '4s,s,4p,p,d,F 2.3,2.3,2.2 2.0',
    'Ge.4e': '4s,s,4p,p,d,F 2.1,2.1,2.1 1.9',
    'Pd.10e': '5s,s,5p,p,4d,d,F 3.1,3.2,3.1 2.9',  # !!!
    'Ag.11e': '5s,s,5p,p,4d,d,F 3.9,3.7,3.6 3.0',  # !!!
    'Cd.12e': '5s,s,5p,p,4d,d,F 2.6,2.4,2.6 2.3',
    'Os.8e': ' 6s,s,6p,p,5d,d,F 2.9,2.9,2.9 2.6',
    'Ir.9e': ' 6s,s,6p,p,5d,d,F 2.9,2.9,2.9 2.6',
    'Pt.10e': '6s,s,6p,p,5d,d,F 2.9,2.9,2.9 2.6',
    'Au.11e': '6s,s,6p,p,5d,d,F 2.7,2.7,2.7 2.5',
    'Hg.12e': '6s,s,6p,p,5d,d,F 2.6,2.5,2.8 2.3',
    'Bi.5e': '6s,s,6p,p,d,F 2.8,2.8,2.8 2.5'}

PARAMS6 = {
    'La': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Ce': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Pr': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Pr.a': '5s,6s,5p,6p,5d,d,4f,f,G     2.1,2.1,2.1,2.1 2.0',
    'Pr.b': '5s,6s,5p,6p,5d,d,4f,f,G     2.1,2.1,2.1,2.1 1.9',
    'Pr.c': '5s,6s,5p,6p,5d,d,4f,f,G     2.2,2.2,2.2,2.2 2.0',
    'Nd': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Pm': '5s,6s,5p,6p,5d,d,4f,f,G       3.4,2.3,2.0,2.1 1.8',
    'Sm': '5s,6s,5p,6p,5d,d,4f,f,G       3.5,2.0,2.1,2.2 1.7',
    'Eu': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Gd': '5s,6s,5p,6p,5d,2.0d,4f,0.5f,G 3.5,3.0,2.8,2.5 2.2',
    'Gd.a': '5s,6s,5p,6p,5d,d,4f,f,G     2.2,2.2,2.2,2.2 2.0',
    'Gd.b': '5s,6s,5p,6p,5d,d,4f,f,G     2.1,2.1,2.1,2.1 2.0',
    'Gd.c': '5s,6s,5p,6p,5d,d,4f,f,G     2.2,2.2,2.2,2.2 1.9',
    'Tb': '5s,6s,5p,6p,5d,d,4f,f,G       2.2,2.2,2.2,2.2 2.0',
    'Dy': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Ho': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Er': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Tm': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Yb': '5s,6s,5p,6p,5d,d,4f,f,G       2.1,2.1,2.1,2.1 2.0',
    'Lu': '5s,6s,5p,6p,5d,d,4f,f,G       2.2,2.2,2.2,2.2 2.0'}


def create(symbol, conf, xc='PBE'):
    proj, rcs, *r0 = conf.split()
    rcs = [float(s) for s in rcs.split(',')]
    if r0:
        r0, = r0
        r0 = float(r0)
    else:
        r0 = 0.9 * min(rcs)

    with open('gen.out', 'w') as fd:
        gen = generate(
            symbol,
            None,
            proj,
            rcs,
            r0,
            None,
            5 if proj[-1].isupper() else 3,
            xc=xc,
            pseudize=('poly', 5),
            ecut=-0.4,
            output=fd)
        try:
            ok = gen.check_all()
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            return False
        setup = gen.make_paw_setup()
        setup.write_xml()
    return True


def create_all(xc='PBE'):
    from contextlib import chdir

    for name, conf in PARAMS9.items():
        if '.' not in name:
            pass  # continue
        dir = Path(name)
        symbol = name.split('.')[0]
        if (dir / f'{symbol}.{xc}').is_file():
            continue
        print(name, conf)
        dir.mkdir(exist_ok=True)
        with chdir(dir):
            ok = create(symbol, conf, xc)
            assert ok


def read_paw_pot(path):
    from xml.etree import ElementTree as ET
    root = ET.parse(path).getroot()
    states = root.find('valence_states')
    r = {}
    for state in states:
        r[int(state.get('l'))] = float(state.get('rc'))
    v0 = root.find('zero_potential')
    return v0.get('type'), float(v0.get('r0')), [rl for _, rl
                                                 in sorted(r.items())]


def analyse(folders):
    results = {}
    data = defaultdict(list)
    for f in folders:
        y = [f.name]
        try:
            ecuts, efcc, de, ecuts_tot, ecuts_diff = convergence_check(f)
        except FileNotFoundError:
            y += [np.nan, np.nan, np.nan]
        else:
            y += [efcc[0] - efcc[-1], ecuts_tot[2], ecuts_diff[2]]
        maxabs, dea, strains = check(f)
        symbol = f.name.split('.')[0]
        try:
            type, r0, rcs = read_paw_pot(f / f'{symbol}.PBE')
        except (FileNotFoundError, TypeError):
            type, r0, rcs = 'old', 0.0, []
        try:
            d = json.loads((f / 'result.json').read_text())
            delta = d['strain']
        except FileNotFoundError:
            delta = np.nan
        data[symbol].append(y + [maxabs, dea, f, type, r0, rcs, delta])
        results[f.name] = (y[2], y[3], strains, delta)
    Path('results.json').write_text(
        json.dumps(results))
    x = []
    y = []
    z = []
    for symbol in chemical_symbols:
        if symbol not in data:
            continue
        best = np.inf
        D = sorted(data[symbol], key=lambda x: (x[0], x[6]))
        for name, de, ec, ecd, maxabs, dea, f, type, r0, rcs, delta in D:
            R = ','.join(f'{r:.1f}' for r in rcs)
            print(f'{name:7} {de:9.6f} {ecd:6.1f} {ec:6.1f} '
                  f'{maxabs:7.4f} {dea:7.4f} {delta:7.4f} '
                  f'{R} {r0:.1f} {type}', f)
            Z = atomic_numbers[symbol]
            x.append(Z)
            y.append([de, ecd, ec, maxabs, dea])
            best = min(best, maxabs)
        z.append(best)
    print(sum(z) / len(z), len(z))
    return x, np.array(y)


def acwf_plot():
    p = Path.home() / 'nh/H-U/old'
    D = []
    for f in p.glob('*/'):
        if '.' in f.name:
            continue
        Z = atomic_numbers[f.name]
        # if Z <= 56 or 72 <= Z <= 83:
        if 84 <= Z <= 96:
            maxabs, dea, d = check(f)
            D += d
    D.sort()
    n = len(D)
    print(n)
    V = [(1 + x)**3 - 1 for x in D]
    n1 = n // 4
    n2 = n // 2
    n3 = n * 3 // 4
    Q1 = (V[n1] + V[n1 + 1]) / 2
    Q2 = (V[n2] + V[n2 + 1]) / 2
    Q3 = (V[n3] + V[n3 + 1]) / 2
    IQR = Q3 - Q1
    print(Q1 - 1.5 * IQR, Q1, Q2, Q3, Q3 + 1.5 * IQR)


def oldvsnew():
    x1, y1 = analyse(list(Path('old').glob('*/')))
    x2, y2 = analyse(list(Path('4').glob('*/')))
    x3, y3 = analyse(list(Path('4-opt-500b').glob('*/')))
    x4, y4 = analyse(list(Path('4-opt-500b-plus').glob('*/')))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    gs = fig.add_gridspec(5, hspace=0)
    axes = gs.subplots(sharex=True)
    # fig, axes = plt.subplots(nrows=5, sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(x1, np.log10(abs(y1[:, i])), 'o-')
        ax.plot(x2, np.log10(abs(y2[:, i])), 'x-')
        ax.plot(x3, np.log10(abs(y3[:, i])), 's-')
        ax.plot(x4, np.log10(abs(y4[:, i])), '+-')
    for ax in axes:
        ax.label_outer()
    plt.show()


def optimize(f, x):
    y = f(tuple(x))
    while True:
        D = []
        for i, xi in enumerate(x):
            for d in [-1, 1]:
                if i > 0 and xi + d == 0:
                    continue
                x2 = list(x)
                x2[i] = xi + d
                y2 = f(tuple(x2))
                D.append((y2, x2))
        y2, x2 = min(D)
        if y2 >= y:
            return x, y
        y = y2
        x = x2


def opt(ecut=600):
    setup_paths[:0] = ['.']
    name = Path.cwd().name
    symbol = name.split('.')[0]
    conf = PARAMS[name]
    proj, rcs, *r0 = conf.split()
    rcs = [float(s) for s in rcs.split(',')]
    if r0:
        r0, = r0
        r0 = float(r0)
    else:
        r0 = round(0.9 * min(rcs), 1)

    if 'f' in proj:
        n = 5
    elif 'd' in proj:
        n = 4
    else:
        n = 3

    @cache
    def f(x):
        ok = None
        if world.rank == 0:
            r0x = x[0] * 0.1
            R = ','.join(f'{xi * 0.1:.1f}' for xi in x[1:])
            conf = f'{proj} {R} {r0x:.1f}'
            ok = create(symbol, conf)
        ok = broadcast(ok, comm=world)
        if not ok:
            return np.inf
        de, s = pfunc(symbol, ecut)
        y = abs(de / 0.001) + abs(s) * 10000
        if world.rank == 0:
            print(f'OPT: {conf:15} {de:10.6f} {s:10.6f} {y}')
        # return (de / 0.001)**2 + s**2 * 1000000
        return y

    rcs += [rcs[-1]] * (n - 1 - len(rcs))
    x = [int(round(r * 10))
         for r in [r0] + rcs]

    x, y = optimize(f, x)
    if world.rank == 0:
        print(x, y)
    f.cache_clear()
    f(tuple(x))
    for name in acwf_structures:
        eos(name, symbol)
    ecuts_check(symbol)


def pfunc(symbol, ecut=600):
    name = 'FCC'
    kwargs = dict(
        xc='PBE',
        occupations=dict(name='fermi-dirac', width=0.0612),
        convergence=dict(density=1e-6),
        kpts=MonkhorstPack(density=3.0),
        experimental={'backwards_compatible': False})
    atoms = reference_structure(symbol, name)
    dft = DFT(
        atoms,
        mode=PW(ecut),
        **kwargs,
        txt='ecuts.txt',
        converge=False)
    try:
        dft.converge()
    except (KohnShamConvergenceError, AssertionError, np.linalg.LinAlgError):
        return np.inf, np.inf
    e = dft.calculate_energy()
    params = Parameters(**kwargs, mode=PW(1200))
    dft = dft.new(atoms, params, log=dft.log)
    try:
        dft.converge()
    except (KohnShamConvergenceError, AssertionError, np.linalg.LinAlgError):
        return np.inf, np.inf
    e0 = dft.calculate_energy()

    kwargs['kpts'] = MonkhorstPack(density=6.0)
    name = 'Diamond'
    atoms = reference_structure(symbol, name)
    try:
        dft = DFT(
            atoms,
            mode=PW(1200),
            **kwargs,
            txt='stress.txt')
    except (KohnShamConvergenceError, AssertionError, np.linalg.LinAlgError):
        return np.inf, np.inf
    s = dft.calculate_stress()
    return e - e0, s[0]


def plotrc():
    r0 = []
    rc = [[], [], [], []]
    for p in PARAMS8.values():
        _, r1, r2 = p.split()
        r0.append(float(r2))
        for l, r in enumerate(r1.split(',')):
            rc[l].append(float(r))
        for l in range(l + 1, 4):
            rc[l].append(np.nan)
    import matplotlib.pyplot as plt
    for l, y in enumerate(rc):
        plt.plot(y, label=str(l))
    plt.plot(r0)
    plt.legend()
    plt.show()


def workflow():
    from myqueue.workflow import run
    run(function=opt, cores=24, tmax='1d')


if __name__ == '__main__':
    import sys
    if 0:
        oldvsnew()
    elif 0:
        acwf_plot()
    elif 0:
        plotrc()
    elif len(sys.argv) == 1:
        # create_all('LDA')
        create_all('PBE')
    elif 0:
        create(sys.argv[1], sys.argv[2])
    else:
        analyse((Path(a) for a in sys.argv[1:]))
