#include "debug_system.hpp"
#include "event_manager.hpp"
#include "gpu_core.hpp"

namespace gpaw
{

// Kernel that just wastes time
__global__ static void wait_kernel(uint64_t wait_cycles)
{
    unsigned long long start = clock64();
    while (clock64() - start < wait_cycles)
    {
        // Do nothing
    }
}

DebugSystem& DebugSystem::get()
{
    static DebugSystem instance;
    return instance;
}

bool DebugSystem::is_system_enabled()
{
    return DebugSystem::get().is_enabled();
}

void DebugSystem::toggle(bool enabled)
{
    this->enabled = enabled;
    if (!enabled)
    {
        // Stop tracking all events. Take a copy of known EventIDs just to be sure that we don't mess up with iterators
        std::vector<EventID> events;
        events.reserve(buffers_for_event.size());

        for (auto kv : buffers_for_event)
        {
            events.push_back(kv.first);
        }

        for (const EventID event : events)
        {
            stop_tracking_event(event);
        }
    }
}

bool DebugSystem::is_enabled() const
{
    return enabled;
}

void DebugSystem::start_tracking_event(EventID id)
{
    // Does not modify Python stuff directly, but we don't want this to be called from many threads. So restrict usage to Python's current thread
    ASSERT_GIL();

    if (!is_enabled()) return;

    assert(EventManager::get().is_valid_event(id));

    std::vector<PyObject*> objects_to_track = EventManager::get().get_event_pyobjects(id);
    if (objects_to_track.empty())
    {
        // No buffers to track
        return;
    }

    // TODO python error handling

    std::vector<PythonBuffer> buffers;
    buffers.reserve(objects_to_track.size());

    for (PyObject* obj : objects_to_track)
    {
        PythonBuffer buf;
        buf.owner = obj;

        void* data_ptr = gpaw::Array_DATA<void>(obj);
        assert(data_ptr);
        buf.ptr = reinterpret_cast<uintptr_t>(data_ptr);
        buf.dtype = dtype_from_typenum(gpaw::Array_TYPE(obj));
        buffers.push_back(buf);

        // PyObject* -> EventID map
        owner_to_events.try_emplace(buf.owner, std::unordered_set<EventID>());
        owner_to_events.at(buf.owner).insert(id);

        // uintptr_t -> EventID map
        buffer_to_events.try_emplace(buf.ptr, std::unordered_set<EventID>());
        buffer_to_events.at(buf.ptr).insert(id);
    }

    // EventID -> PythonBuffers map
    buffers_for_event.emplace(id, buffers);
}

void DebugSystem::stop_tracking_event(EventID id)
{
    // Same threading logic as in start_tracking_event
    ASSERT_GIL();

    if (!is_enabled()) return;

    if (!map_contains(buffers_for_event, id))
    {
        return;
    }

    for (const PythonBuffer& buffer : buffers_for_event.at(id))
    {
        // Check that the PyObject data pointer is still the same as when we registered it.

        /* FIXME this needs more thought. Fails immediately if the 'owner' has already died,
        which can happen if the event system does not take ownership problem in itself.
        And the check itself seems restrictive because do we know that Cupy/Numpy objects never change their data ptr?
        */

        /*
        void* data_ptr = gpaw::Array_DATA<void>(buffer.owner);
        assert(data_ptr);
        const uintptr_t ptr_current = reinterpret_cast<uintptr_t>(data_ptr);
        assert(ptr_current == buffer.ptr && "PyObject <-> data_ptr mapping broken, pointer has changed?");
        */

        // Remove event from PyObject* -> EventID and uintptr_t -> EventID maps
        PyObject* owner = buffer.owner;
        assert(map_contains(owner_to_events, owner));
        owner_to_events.at(owner).erase(id);
        if (owner_to_events[owner].empty())
        {
            owner_to_events.erase(owner);
        }

        const uintptr_t ptr = buffer.ptr;
        assert(map_contains(buffer_to_events, ptr));
        buffer_to_events.at(ptr).erase(id);
        if (buffer_to_events[ptr].empty())
        {
            buffer_to_events.erase(ptr);
        }
    }

    buffers_for_event.erase(id);

    // Remove from list of finished events
    UniqueLock lock(finished_events_mutex);
    auto it = finished_events.find(id);
    if (it != finished_events.end())
    {
        finished_events.erase(id);
    }
}

bool DebugSystem::is_event_finished(EventID id)
{
    // Check first our internal cache of finished events.
    // It may be more up-to-date than the global manager due to async debug callbacks
    {
        SharedLock lock(finished_events_mutex);
        if (finished_events.find(id) != finished_events.end())
        {
            return true;
        }
    }
    // Then check through the event manager
    const bool has_finished = EventManager::get().has_event_finished(id);
    if (has_finished)
    {
        // Update debug system's cache
        UniqueLock lock(finished_events_mutex);
        finished_events.emplace(id);
    }

    return has_finished;
}

void DebugSystem::internal_mark_event_finished(EventID id)
{
    if (!is_enabled() || !map_contains(buffers_for_event, id))
    {
        return;
    }

    UniqueLock lock(finished_events_mutex);
    finished_events.emplace(id);
}

bool DebugSystem::is_event_tracked(EventID id) const
{
    if (!is_enabled())
    {
        return false;
    }

    return map_contains(buffers_for_event, id);
}

bool DebugSystem::is_pointer_tracked(uintptr_t ptr) const
{
    if (!is_enabled())
    {
        return false;
    }

    return map_contains(buffer_to_events, ptr);
}

bool DebugSystem::is_object_tracked(PyObject* obj) const
{
    if (!is_enabled())
    {
        return false;
    }

    return map_contains(owner_to_events, obj);
}

bool DebugSystem::ensure_object_not_in_use(PyObject* obj)
{
    if (!is_object_tracked(obj))
    {
        return true;
    }

    std::vector<EventID> events;
    for (auto event : owner_to_events.at(obj))
    {
        if (!is_event_finished(event))
        {
            events.push_back(event);
        }
    }

    if (events.size() > 0)
    {
        printf("Error, object is in use by %zu unfinished events\n", events.size());
        print_event_info(events);
        fflush(stdout);
        return false;
    }

    return true;
}

bool DebugSystem::ensure_pointer_not_in_use(uintptr_t ptr)
{
    if (!is_pointer_tracked(ptr))
    {
        return true;
    }

    std::vector<EventID> events;
    for (EventID event : buffer_to_events.at(ptr))
    {
        if (!is_event_finished(event))
        {
            events.push_back(event);
        }
    }

    if (events.size() > 0)
    {
        printf("Error, pointer is in use by %zu unfinished events\n", events.size());
        print_event_info(events);
        fflush(stdout);
        return false;
    }

    return true;
}

void DebugSystem::configure_stream_delay(double microseconds)
{
    stream_delay_microseconds = microseconds;
}

void DebugSystem::insert_stream_delay(gpuStream_t stream)
{
    if (stream_delay_microseconds > 0)
    {
        launch_wait_kernel(stream, stream_delay_microseconds);
    }
}

void DebugSystem::print_event_info(EventID id) const
{
    printf("id : %zu\n", (size_t)id);
}

void DebugSystem::print_event_info(const std::vector<EventID>& events) const
{
    for (EventID id : events)
    {
        print_event_info(id);
    }
}

void DebugSystem::launch_wait_kernel(gpuStream_t stream, double microseconds)
{
    // Convert time to cycles based on clock frequency

    int device = get_current_device();
    gpuDeviceProp prop;
    gpuGetDeviceProperties(&prop, device);

    const int clock_rate_khz = prop.clockRate;
    const uint64_t wait_cycles = microseconds / 1000.0 * (uint64_t)clock_rate_khz;

    launch_kernel(wait_kernel, 1, 1, 0, stream, wait_cycles);
}

}
