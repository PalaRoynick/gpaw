#include "gpu/cupy_interface/device_array.hpp"
#include "lfc_gpu_impl.cuh"
#include "lfc_gpu.cuh"

namespace py = pybind11;

LFCGpu::~LFCGpu()
{
    namespace gpu = gpaw::gpu;

    // Free PhiDataGpu
    gpu::Tfree(phi_data.spline_idx_i);
    gpu::Tfree(phi_data.first_mu_i);
    gpu::Tfree(phi_data.l_i);
    gpu::Tfree(phi_data.position_iv);

    // Free BlockDataGpu
    gpu::Tfree(block_data.all_blocks_phi_idx_j);
    gpu::Tfree(block_data.start_j);
    gpu::Tfree(block_data.block_start_bc);

    // Free PhiPairDataGpu
    gpu::Tfree(phi_pair_data.block_idx_p);
    gpu::Tfree(phi_pair_data.phi_idx_r_p);
    gpu::Tfree(phi_pair_data.phi_idx_s_p);
}

void LFCGpu::construct_density(gpaw::PyDeviceArray &nt_G, const gpaw::PyDeviceArray &rho_MM, size_t mu_start, size_t mu_end) const
{
    if (!nt_G.c_contiguous || nt_G.ndim() != 3)
    {
        throw std::invalid_argument("nt_G must be 3D and C-contiguous");
    }
    if (!rho_MM.c_contiguous || rho_MM.ndim() != 2)
    {
        throw std::invalid_argument("rho_MM must be 2D and C-contiguous");
    }
    if (!nt_G.dtype.is(rho_MM.dtype))
    {
        throw std::invalid_argument("dtype mismatch: nt_G and rho_MM must have same dtype");
    }

    const py::dtype dtype = nt_G.dtype;
    switch (dtype.normalized_num())
    {
    case py::dtype::num_of<float>():
        LFCGpuImpl::launch_construct_density<float>(*this, nt_G, rho_MM, mu_start, mu_end); break;
    case py::dtype::num_of<double>():
        LFCGpuImpl::launch_construct_density<double>(*this, nt_G, rho_MM, mu_start, mu_end); break;
    default:
        throw std::runtime_error("Unsupported dtype");
    }

}

void LFCGpu::add_to_density(gpaw::PyDeviceArray &nt_sG, const gpaw::PyDeviceArray &f_sM) const
{
    if (!nt_sG.c_contiguous || nt_sG.ndim() != 4)
    {
        throw std::invalid_argument("nt_sG must be 4D and C-contiguous");
    }
    if (!f_sM.c_contiguous || f_sM.ndim() != 2)
    {
        throw std::invalid_argument("f_sM must be 2D and C-contiguous");
    }
    if (!nt_sG.dtype.is(f_sM.dtype))
    {
        throw std::invalid_argument("dtype mismatch: nt_sG and f_sM must have same dtype");
    }
    if (nt_sG.shape[0] != f_sM.shape[0])
    {
        throw std::invalid_argument("Shape mismatch: nt_sG and f_sM first dimensions must match");
    }

    const py::dtype dtype = nt_sG.dtype;
    switch (dtype.normalized_num())
    {
    case py::dtype::num_of<float>():
        LFCGpuImpl::launch_add_to_density<float>(*this, nt_sG, f_sM); break;
    case py::dtype::num_of<double>():
        LFCGpuImpl::launch_add_to_density<double>(*this, nt_sG, f_sM); break;
    default:
        throw std::runtime_error("Unsupported dtype");
    }
}

void LFCGpu::calculate_potential_matrix(const gpaw::PyDeviceArray& vt_G, gpaw::PyDeviceArray& out_V_MN,
                                        size_t mu_start, size_t mu_end) const
{
    if (!vt_G.c_contiguous || vt_G.ndim() != 3)
    {
        throw std::invalid_argument("vt_G must be 3D and C-contiguous");
    }
    if (!out_V_MN.c_contiguous || out_V_MN.ndim() != 2)
    {
        throw std::invalid_argument("out_V_MN must be 2D and C-contiguous");
    }
    if (!vt_G.dtype.is(out_V_MN.dtype))
    {
        throw std::invalid_argument("dtype mismatch: vt_G and out_V_MN must have same dtype");
    }

    const py::dtype dtype = vt_G.dtype;
    switch (dtype.normalized_num())
    {
    case py::dtype::num_of<float>():
        LFCGpuImpl::launch_calculate_potential_matrix_phi_pairs<float>(*this, vt_G, out_V_MN, mu_start, mu_end); break;
    case py::dtype::num_of<double>():
        LFCGpuImpl::launch_calculate_potential_matrix_phi_pairs<double>(*this, vt_G, out_V_MN, mu_start, mu_end); break;
    default:
        throw std::runtime_error("Unsupported dtype");
    }
}
