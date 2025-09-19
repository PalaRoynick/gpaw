#include "events_python_interface.h"
#include "event_manager.hpp"

CLINKAGE PyObject* destroy_expired_events(PyObject* self, PyObject* args)
{
    gpaw::EventManager::get().process_finished_events();
    Py_RETURN_NONE;
}
