#include "event_manager.hpp"

namespace gpaw
{

EventManager<true> g_multithreaded_manager;
EventManager<false> g_event_manager;

} // namespace gpaw