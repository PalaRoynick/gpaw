import numpy as np

size_c = (2, 3, 4)

# Old style: produces non-c-contiguous tmp_R
shape = (size_c[0], size_c[1], size_c[2] // 2 + 1)
tmp_Q = np.empty(shape, np.complex128)
tmp_R = tmp_Q.view(np.float64)[:, :, :size_c[2]]
print(f"Old: {tmp_R.flags.c_contiguous=}, {tmp_Q.flags.c_contiguous=}, {tmp_R.shape=}, {tmp_Q.shape=}")

padded_real_size = (size_c[0], size_c[1], 2 * (size_c[2] // 2 + 1))
real_block = np.empty(padded_real_size, np.float64)
tmp_R = real_block[:, :, :size_c[2]]
tmp_Q = real_block.view(dtype=np.complex128)

print(f"New: {tmp_R.flags.c_contiguous=}, {tmp_Q.flags.c_contiguous=}, {tmp_R.shape=}, {tmp_Q.shape=}")
