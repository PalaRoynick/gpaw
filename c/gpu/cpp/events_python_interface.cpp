#include "events_python_interface.h"
#include "event_manager.hpp"
#include "debug_system.hpp"
#include "events.hpp"
#include "utils.hpp"

#include <vector>
#include <string>

PyObject* DebugSystemError = nullptr;

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

CLINKAGE PyObject* get_overlapping_events_as_string(PyObject* self, PyObject* args)
{
    long long start;
    long long nbytes;
    if (!PyArg_ParseTuple(args, "LL", &start, &nbytes))
    {
        return NULL;
    }

    gpaw::MemoryBlock block;
    block.start = static_cast<uintptr_t>(start);
    block.nbytes = static_cast<size_t>(nbytes);
    std::vector<gpaw::EventID> events = gpaw::DebugSystem::get().get_overlapping_events(block);

    // Return as a list of entries like {"EventID": 1, name: "something", ...}
    std::vector<std::string> event_strings = gpaw::DebugSystem::get().make_debug_strings(events);

    PyObject* list_obj = gpaw::as_python_list(event_strings);

    // Is already NULL if there was an exception
    return list_obj;
}

/* This packs the error metadata into a dict and propagates it to Python as a custom exception.
Can access the dict as "e.args[0]". Would be better to make the metadata fields direct attributes of the exception,
but this requires too much boilerplate in the Python.h API. */
static void raise_debug_system_error(const gpaw::MemoryBlock& block, const std::vector<gpaw::EventID>& overlapping_events)
{
    using gpaw::EventID;

    std::vector<std::string> event_strings = gpaw::DebugSystem::get().make_debug_strings(overlapping_events);
    PyObject* list_obj = gpaw::as_python_list(event_strings);

    if (!list_obj)
    {
        // Likely OOM
        PyErr_SetString(PyExc_RuntimeError, "DebugSystem detected an error but failed to propagate payload to Python");
        return;
    }

    PyObject *payload = Py_BuildValue("{s:s, s:k, s:n, s:O}",
        "reason", "Data race with ongoing events",
        "address", (unsigned long long)block.start,
        "length",  (unsigned long long)block.nbytes,
        "events", list_obj
    );
    Py_DECREF(list_obj);

    if (!payload)
    {
        PyErr_SetString(PyExc_RuntimeError, "DebugSystem detected an error but failed to propagate payload to Python");
        return;
    }

    PyErr_SetObject(DebugSystemError, payload);
    Py_DECREF(payload);
}


bool ensure_no_async_overlap(PyObject* arr, bool fatal)
{
    using gpaw::DebugSystem;

    if (!DebugSystem::is_system_enabled())
    {
        return true;
    }

    void* data_ptr = gpaw::Array_DATA<void>(arr);
    if (!data_ptr)
    {
        // No data, so can't overlap
        return true;
    }

    gpaw::MemoryBlock block;
    block.start = reinterpret_cast<uintptr_t>(data_ptr);
    block.nbytes = static_cast<size_t>(gpaw::Array_NBYTES(arr));

    DebugSystem& debug_system = DebugSystem::get();
    std::vector<gpaw::EventID> events = debug_system.get_overlapping_events(block);

    if (events.empty())
    {
        return true;
    }

    raise_debug_system_error(block, events);
    if (!fatal && PyErr_Occurred())
    {
        PyErr_Print();
    }
    return false;
}
