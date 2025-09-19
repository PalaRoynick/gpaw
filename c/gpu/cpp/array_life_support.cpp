#include "array_life_support.hpp"
#include "pyarray_utils.hpp"
#include "utils.hpp"
#include "gpu_core.hpp"
#include "event_manager.hpp"

#include <Python.h>
#include <mutex>

namespace gpaw
{

namespace life_support
{
// Global cache for storing objects that have pending unpin/decref
static std::vector<PyObject*> g_pending_decrefs; // should this be volatile?!
static std::mutex g_pending_decrefs_mutex;

} // namespace life_support


PyObjectPinner::PyObjectPinner()
{
}

PyObjectPinner::PyObjectPinner(size_t reserve_count)
{
    objects.reserve(reserve_count);
}

void PyObjectPinner::commit()
{
#ifdef GPAW_GPU_ARRAY_DEBUG
    assert(!has_committed && "Can't commit object pinning twice");
    has_committed = true;
#endif

    for (PyObject* obj : objects)
    {
        Py_INCREF(obj);
    }
}

void PyObjectPinner::schedule_unpin(gpuStream_t stream)
{

#ifdef GPAW_GPU_ARRAY_DEBUG
    assert(has_committed && "You are calling schedule_unpin() without committing the pinning first");
#endif

    if (objects.empty())
    {
        return;
    }

    // Move the stored pointers to a lambda so the callback works even if this calling object has been destroyed
    auto wrapper = [vec_copy = std::move(objects)]() mutable
    {
        // Add our pointers to the global cache of pending unpins
        std::lock_guard<std::mutex> lock(life_support::g_pending_decrefs_mutex);
        for (PyObject* obj : vec_copy)
        {
            life_support::g_pending_decrefs.push_back(obj);
        }
    };

    gpu_host_callback(stream, wrapper);
}

} // namespace gpaw
