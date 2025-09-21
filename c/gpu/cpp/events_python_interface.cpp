#include "events_python_interface.h"
#include "event_manager.hpp"
#include "debug_system.hpp"

CLINKAGE PyObject* destroy_expired_events(PyObject* self, PyObject* args)
{
    gpaw::EventManager::get().process_finished_events();
    Py_RETURN_NONE;
}

CLINKAGE PyObject* is_debug_system_enabled(PyObject* self, PyObject* args)
{
    const bool enabled = gpaw::DebugSystem::is_system_enabled();
    return PyBool_FromLong(static_cast<long>(enabled));
}

CLINKAGE PyObject* toggle_debug_system(PyObject* self, PyObject* args)
{
    int should_enable;
    if (!PyArg_ParseTuple(args, "p", &should_enable))
    {
        return NULL;
    }

    gpaw::DebugSystem::get().toggle(static_cast<bool>(should_enable));
    Py_RETURN_NONE;
}

CLINKAGE PyObject* configure_debug_stream_delay_microseconds(PyObject* self, PyObject* args)
{
    double microsecs;
    if (!PyArg_ParseTuple(args, "d", &microsecs))
    {
        return NULL;
    }

    gpaw::DebugSystem::get().configure_stream_delay(microsecs);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* configure_events_take_pyobject_ownership(PyObject* self, PyObject* args)
{
    int take_ownership;
    if (!PyArg_ParseTuple(args, "p", &take_ownership))
    {
        return NULL;
    }
    gpaw::EventManager::get().toggle_events_take_pyobject_ownership(take_ownership);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* ensure_buffer_not_in_use(PyObject* self, PyObject* args)
{
    long long ptr;
    if (!PyArg_ParseTuple(args, "L", &ptr))
    {
        return NULL;
    }

    // FIXME: Actually this cast may not work 32bit platforms. Should check sizeof(uintptr_t) >= sizeof(long long).
    // But better would be to just parse it as proper type...

    const bool not_in_use = gpaw::DebugSystem::get().ensure_pointer_not_in_use(static_cast<uintptr_t>(ptr));
    return PyBool_FromLong(static_cast<long>(not_in_use));
}

CLINKAGE PyObject* ensure_object_not_in_use(PyObject* self, PyObject* args)
{
    PyObject* obj;
    if (!PyArg_ParseTuple(args, "O", &obj))
    {
        return NULL;
    }

    const bool not_in_use = gpaw::DebugSystem::get().ensure_object_not_in_use(obj);
    return PyBool_FromLong(static_cast<long>(not_in_use));
}

CLINKAGE PyObject* ensure_block_not_in_use(PyObject* self, PyObject* args)
{
    long long start;
    long long nbytes;
    if (!PyArg_ParseTuple(args, "LL", &start, &nbytes))
    {
        return NULL;
    }

    const bool not_in_use = gpaw::DebugSystem::get().ensure_block_not_in_use(static_cast<uintptr_t>(start), static_cast<size_t>(nbytes));
    return PyBool_FromLong(static_cast<long>(not_in_use));
}
