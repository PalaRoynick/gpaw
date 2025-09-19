#pragma once

#include "events.hpp"
#include "pyarray_utils.hpp"

#include <cstdint>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>

// TODO Bind global funcs like these to Python, make them access the singleton
//void toggle_debug_mode(bool enabled);
//bool is_debug_mode();

namespace gpaw
{

struct PythonBuffer
{
    PyObject* owner;
    uintptr_t ptr;
    DataType dtype;
};

// Singleton
class DebugSystem
{
public:
    static DebugSystem& get();
    static bool is_system_enabled();

    void toggle(bool enabled);
    bool is_enabled() const;

    void start_tracking_event(EventID id);
    void stop_tracking_event(EventID id);
    bool is_event_finished(EventID id);
    // FIXME should be private. But need to refactor event creation "hooking"
    void internal_mark_event_finished(EventID id);

protected:

    bool enabled = true;

    // Manual "bidirectional map" setup for buffer lookups

    // Snapshot of buffers as they were when event tracking started.
    std::unordered_map<EventID, std::vector<PythonBuffer>> buffers_for_event;
    std::unordered_map<PyObject*, std::unordered_set<EventID>> owner_to_events;
    std::unordered_map<uintptr_t, std::unordered_set<EventID>> buffer_to_events;

    /* List of tracked events that have finished. We store this info here
    instead of modifying the events directly, because the DebugSystem can do
    async callbacks eg. from a GPU stream to signal a finished kernel event.
    Async event modifications through the EventManager would require the manager
    to be thread-safe, which has performance implications even when the debug system
    is disabled.
    */
    std::unordered_set<EventID> finished_events;

    using UniqueLock = std::unique_lock<std::shared_mutex>;
    using SharedLock = std::shared_lock<std::shared_mutex>;

    std::shared_mutex finished_events_mutex;

private:
    // singleton pattern
    DebugSystem() {}
    ~DebugSystem() {}
    DebugSystem& operator=(const DebugSystem&) = delete;
    DebugSystem(const DebugSystem&) = delete;

};

/* Probably easiest approach: in all "create event" functions of the event manager,
inform the DebugSystem about the newly created event and which object it's using.
DebugSystem can then store map of objects -> EventID. Then cupy memory hooks can ask the DebugSystem if a given object is used in any existing event. */


} // namespace gpaw
