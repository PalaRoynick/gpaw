#include "events.hpp"
#include <Python.h>

namespace gpaw
{

EventManager<true> g_multithreaded_manager;
EventManager<false> g_event_manager;

KernelEvent::~KernelEvent()
{
    for (PyObject* obj : used_objects)
    {
        // would Py_CLEAR be safer?
        assert(obj && "Null PyObject, can't happen since we should have ownership");
        Py_DECREF(obj);
    }
    gpuEventDestroy(native_event);
}

bool KernelEvent::has_finished() const
{
    if (cached_has_finished)
    {
        return true;
    }

    const gpuError_t status = gpuEventQuery(native_event);
    // FIXME not quite robust as this can also return error codes from previous async launches
    if (status == gpuSuccess)
    {
        cached_has_finished = true;
        return true;
    }
    else if (status == gpuErrorNotReady)
    {
        return false;
    }
    else
    {
        printf("gpuEventQuery error: %s\n", gpuGetErrorString(status));
        fflush(stdout);
        gpaw_set_runtime_error(gpuGetErrorString(status));
        return false;
    }
}

void KernelEvent::on_start()
{
    for (PyObject* obj : used_objects)
    {
        Py_INCREF(obj);
    }
}

} // namespace gpaw
