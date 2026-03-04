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
    void add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const;


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

// PROBLEM: can't partially specialize a function AFAIK, so will need to put this inside a class probs.
template<>
std::unique_ptr<LFCBase> build_impl<double, false>(GridInfo&& grid, std::vector<GridBlock>&& blocks, PhiData&& phi_data)
{
    std::unique_ptr<LFCCpu<double>> lfc = std::make_unique<LFCCpu<double>>();
    assert(lfc != nullptr);

    lfc->blocks = std::move(blocks);
    lfc->phi_data = std::move(phi_data);

    //lfc->gpu_impl = std::move(lfc->upload_to_gpu());

    return std::move(lfc);
}

}
