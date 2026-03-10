import numpy as np


def get_ndim(ibzwfs):

    return ibzwfs.nbands


def get_n_occ(f_n):
    occupied = f_n > 1.0e-10
    n_occ = len(f_n[occupied])

    return n_occ
