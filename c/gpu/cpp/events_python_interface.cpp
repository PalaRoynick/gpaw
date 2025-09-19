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
