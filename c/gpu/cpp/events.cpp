#include "events.hpp"
#include <Python.h>

namespace gpaw
{

KernelEvent::~KernelEvent()
{
    for (PyObject* obj : used_objects)
    {
        // would Py_CLEAR be safer?
        assert(obj && "Null PyObject, can't happen since we should have ownership");
        Py_DECREF(obj);
    }
    if (native_event != nullptr)
    {
        gpuEventDestroy(native_event);
    }
}

KernelEvent::KernelEvent(KernelEvent&& other) noexcept
    : EventBase(std::move(other)),
      native_event(other.native_event),
      stream(other.stream),
      device(other.device),
      cached_has_finished(other.cached_has_finished),
      used_objects(std::move(other.used_objects))
{
    other.used_objects.clear();
    other.native_event = nullptr;
}

KernelEvent& KernelEvent::operator=(KernelEvent&& other) noexcept
{
    if (this != &other)
    {
        EventBase::operator=(std::move(other));

        for (PyObject* obj : used_objects)
        {
            Py_DECREF(obj);
        }
        if (native_event)
        {
            gpuEventDestroy(native_event);
        }

        native_event = other.native_event;
        stream = other.stream;
        device = other.device;
        cached_has_finished = other.cached_has_finished;
        used_objects = std::move(other.used_objects);

        other.used_objects.clear();
        other.native_event = nullptr;
    }

    return *this;
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
