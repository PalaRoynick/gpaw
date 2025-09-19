#include "array_life_support.hpp"
#include "pyarray_utils.hpp"
#include "utils.hpp"
#include "gpu_core.hpp"
#include "event_manager.hpp"

#include <Python.h>
#include <mutex>

namespace gpaw
{

PyObjectPinner::PyObjectPinner()
{
}

PyObjectPinner::PyObjectPinner(size_t reserve_count)
{
    objects.reserve(reserve_count);
}


} // namespace gpaw
