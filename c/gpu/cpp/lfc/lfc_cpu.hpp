#pragma once

#include "lfc_base.hpp"
#include "lfc_builder.hpp"

namespace gpaw
{

template<typename T>
class LFCCpu : public LFCBase
{
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

public:
    static constexpr bool is_single_precision() { return std::is_same_v<T, float>; }

    // TODO fix dtype
    void add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const override;
    void calculate_potential_matrix(const NDArray<c_contig, ndim<3>>& vt_G, NDArray<c_contig, ndim<2>>& out_V_MN) const override {}

    std::vector<GridBlock> blocks;
    PhiData phi_data;
private:

    //std::unique_ptr<LFCGpu<T>> upload_to_gpu();

    //std::unique_ptr<LFCGpu<T>> gpu_impl;

    friend class BasisFunctionCollectionBuilder;
};

template<typename T>
inline void LFCCpu<T>::add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const
{

}

template<typename T>
std::unique_ptr<LFCBase> build_impl_cpu(GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data)
{
    std::unique_ptr<LFCCpu<T>> lfc = std::make_unique<LFCCpu<T>>();
    assert(lfc != nullptr);

    lfc->blocks = std::move(blocks);
    lfc->phi_data = std::move(phi_data);

    throw std::runtime_error("This is WIP");

    return std::move(lfc);
}

}
