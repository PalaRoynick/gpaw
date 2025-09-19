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

    // Enable or disable the debug system
    void toggle(bool enabled);
    bool is_enabled() const;

    void start_tracking_event(EventID id);
    void stop_tracking_event(EventID id);
    bool is_event_finished(EventID id);
    // FIXME should be private. But need to refactor event creation "hooking"
    void internal_mark_event_finished(EventID id);

    bool is_event_tracked(EventID id) const;
    bool is_pointer_tracked(uintptr_t ptr) const;
    bool is_object_tracked(PyObject* obj) const;

    bool ensure_object_not_in_use(PyObject* obj);
    bool ensure_pointer_not_in_use(uintptr_t ptr);

    /* Set amount of delay that the DebugSystem will insert to the GPU stream after every kernel launch
    Default is 200 microseconds. Value of 0 or less will disable the delay insertions. */
    void configure_stream_delay(double microseconds);

    void insert_stream_delay(gpuStream_t stream);

protected:

    void print_event_info(EventID id) const;
    void print_event_info(const std::vector<EventID>& events) const;

    bool enabled = false;

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

    double stream_delay_microseconds = 200;

    // Launches a kernel that just waits for approximately the given time
    void launch_wait_kernel(gpuStream_t stream, double microseconds);
};

} // namespace gpaw
