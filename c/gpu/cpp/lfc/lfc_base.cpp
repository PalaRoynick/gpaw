#include "lfc_base.hpp"

namespace gpaw
{

void BasisFunctionCollection::add_to_density(NDArray<c_contig, ndim<4>>& nt_sG, const NDArray<c_contig, ndim<2>>& f_sM) const
{
    impl->add_to_density(nt_sG, f_sM);
}

void BasisFunctionCollection::calculate_potential_matrix(const NDArray<c_contig, ndim<3>>& vt_G, NDArray<c_contig, ndim<2>>& out_V_MN) const
{
    impl->calculate_potential_matrix(vt_G, out_V_MN);
}

} // namespace gpaw
