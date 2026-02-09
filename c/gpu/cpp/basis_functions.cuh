#pragma once

#include "python_utils.h"
#include "splines/radial_spline_gpu.cuh"
#include "ndarray.hpp"
#include "gpu_core.hpp"
#include <cstdint>
#include <unordered_map>

namespace gpaw
{

using atom_index_t = uint32_t;

struct Atom
{
    uint32_t l;
    // Position in real units
    std::array<double, 3> position;
    TRadialSplineGpu<double> spline; // will be redundant, atoms can share splines
};

template<typename T>
class BasisFunctionCollection
{
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

public:
    BasisFunctionCollection(size_t num_atoms);

    void make_atom();

    void add_to_density(
        NDArray<T, device::gpu, c_contig, ndim<4>>& nt_sG,
        const std::vector<NDArray<T, device::gpu, c_contig, ndim<2>>>& f_asi);

private:

    // we rearrange atom indices to range [0, ... N-1]
    std::unordered_map<atom_index_t, atom_index_t> new_atom_index;

};

template<typename T>
class SplineMap
{
    // Use signed integers where possible (not allowed to overflow -> possibly faster code)

    __device__ int32_t get_num_splines_for_atom(int32_t a) const { return atoms[a].num_splines; }
    __device__ int32_t get_first_spline_idx_for_atom(int32_t a) const { return atoms[a].first_spline_idx; }
    __device__ T get_lattice_spacing(int32_t dir) const { return lattice_spacing[dir]; }
    __device__ int32_t get_num_atoms() const { return num_atoms; }
    __device__ T get_atom_position(int32_t a, int32_t dir) const { return atoms[a].position[dir]; }
    __device__ T evaluate_spline(int32_t spline_idx, T r) const { return splines[spline_idx].evaluate(r); }

private:

    struct AtomData
    {
        int32_t num_splines = 0;
        int32_t first_spline_idx = 0;
        T position_v[3];
    };

    AtomData* GPAW_RESTRICT atoms = nullptr;
    int32_t num_atoms = 0;

    // All splines directly in GPU memory
    TRadialSplineGpu<T>* GPAW_RESTRICT splines = nullptr;

    // Lattice spacing in each direction
    T lattice_spacing[3];
};

template<typename T>
__global__ void add_to_density_kernel(SplineMap<T> spline_map, T* GPAW_RESTRICT nt_sG, T** f_a_si, int32_t num_spins)
{
    // n_s(x) += sum_{a,i} f^a_{si} |phi^a_{j}(x)|^2

    // TODO get grid dimensions from somewhere and ensure thread xyz is valid

    using Vec3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

    const T x = spline_map.get_lattice_spacing(0) * (blockIdx.x * blockDim.x + threadIdx.x);
    const T y = spline_map.get_lattice_spacing(1) * (blockIdx.y * blockDim.y + threadIdx.y);
    const T z = spline_map.get_lattice_spacing(2) * (blockIdx.z * blockDim.z + threadIdx.z);

    for (int32_t a = 0; a < spline_map.get_num_atoms(); ++a)
    {
        // current xyz point relative to the atom
        Vec3 diff;
        diff.x = x - spline_map.get_atom_position(a, 0);
        diff.y = y - spline_map.get_atom_position(a, 1);
        diff.z = z - spline_map.get_atom_position(a, 2);
        const T distance = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);

        // calculate strides to nt_sG (C-style 4D array). Assumes we launched one thread per XYZ:
        const int64_t nx = gridDim.x * blockDim.x;
        const int64_t ny = gridDim.y * blockDim.y;
        const int64_t nz = gridDim.z * blockDim.z;

        const int64_t x_stride = ny * nz;
        const int64_t s_stride = nx * ny * nz;

        const int32_t num_splines_for_atom = spline_map.get_num_splines_for_atom(a);

        for (int32_t local_spline_idx = 0; local_spline_idx < num_splines_for_atom; ++local_spline_idx)
        {
            const int32_t spline_idx = spline_map.get_first_spline_idx_for_atom(a) + local_spline_idx;

            // TODO spherical harmonics. For now, test with just the radial part
            const T phi = spline_map.evaluate_spline(spline_idx, distance);

            for (int32_t s = 0; s < num_spins; ++s)
            {
                const int64_t out_idx = s * s_stride + x * x_stride + y * nz + z;
                const T f = f_a_si[a][s * num_splines_for_atom + local_spline_idx];

                nt_sG[out_idx] += f * phi*phi;
            }
        }
    }
}

template<typename T>
inline void BasisFunctionCollection<T>::add_to_density(
    NDArray<T, device::gpu, c_contig, ndim<4>>& nt_sG,
    const std::vector<NDArray<T, device::gpu, c_contig, ndim<2>>>& in_f_asi)
{
    T** f_a_si = nullptr;
}

} // namespace gpaw
