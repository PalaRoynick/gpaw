import cupy as cp

def add_memory_padding(a: cp.ndarray, front_bytes=1024, back_bytes=1024):
    """"""
    array_start = a.data.mem


A = cp.ones((1, 16, 16, 16))
B = A.sum(axis=(-2, -1))

print(B)