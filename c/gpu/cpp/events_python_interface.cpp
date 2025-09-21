#include "events_python_interface.h"
#include "event_manager.hpp"
#include "debug_system.hpp"
#include "events.hpp"
#include "utils.hpp"

#include <vector>
#include <string>

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

    if (events.empty())
    {
        return PyList_New(0);
    }

    // Return as a list of entries like {"EventID": 1, name: "something", ...}

    PyObject* list_obj = PyList_New(events.size());
    if (!list_obj)
    {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < (Py_ssize_t)events.size(); i++)
    {
        const gpaw::EventID id = events[i];
        auto event_dict = gpaw::EventManager::get().as_string_dict(id);
        // Above we checked that the event had NOT finished yet, but by now it may have.
        // So for this output we don't want the "finished" field because it may be misleading.
        if (gpaw::map_contains(event_dict, std::string("finished")))
        {
            event_dict.erase("finished");
        }

        const std::string event_str = gpaw::to_json_entry(event_dict);

        PyObject* str_obj = PyUnicode_FromString(event_str.c_str());
        if (!str_obj)
        {
            Py_DECREF(list_obj);
            return NULL;
        }

        PyList_SetItem(list_obj, i, str_obj); // steals reference
    }

    return list_obj;
}
