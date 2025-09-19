#pragma once

#include "events.hpp"
#include "debug_system.hpp"

#include <unordered_map>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <variant>
#include <vector>

namespace gpaw
{

using EventVariant = std::variant<KernelEvent>;

// FIXME does not need to be thread safe. Just restrict event management to the Python thread.
// DebugSystem can keep internal track of async stuff
template<bool multithreaded>
class EventManager
{
public:
    EventManager() : next_event_id(0) {}

    /* Creates a new GPU event and records it in the given GPU stream on current device.
    Flags are native cuda or hip event creation flags */
    EventID create_kernel_exec_event(const std::string& name, gpuStream_t stream, std::vector<PyObject*>&& objects, unsigned int flags = gpuEventDefault);

    void process_finished_events();

    bool has_event_finished(EventID id) const;
    // Manually mark an event as finished
    void mark_event_finished(EventID id);
    EventType get_event_type(EventID id) const;
    bool is_valid_event(EventID id) const;

    // Returns PyObjects associated with an event. Will be empty if the event is invalid
    std::vector<PyObject*> get_event_pyobjects(EventID id);

protected:

    // Generates a new globally unique event ID
    EventID generate_unique_id();

    std::unordered_map<EventID, EventVariant> events;

private:

    // Use locks and mutexes if multithreaded, otherwise define dummy types that do nothing
    struct DummyMutex
    {
        void lock() {}
        void unlock() {}
    };
    struct DummyLock
    {
        DummyLock(...) {}
    };

    using MutexType = std::conditional_t<multithreaded, std::shared_mutex, DummyMutex>;
    using UniqueLock = std::conditional_t<multithreaded, std::unique_lock<MutexType>, DummyLock>;
    using SharedLock = std::conditional_t<multithreaded, std::shared_lock<MutexType>, DummyLock>;

    // Mutex for locking the events container
    mutable MutexType mutex;

    // Counter for generating event IDs. Must be atomic in thread safe mode
    std::conditional_t<multithreaded, std::atomic<EventID>, EventID> next_event_id;
};

/******* Templated implementations */

template<bool multithreaded>
EventID EventManager<multithreaded>::generate_unique_id()
{
    if constexpr (multithreaded)
    {
        return next_event_id.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        return next_event_id++;
    }
}

template<bool multithreaded>
EventID EventManager<multithreaded>::create_kernel_exec_event(const std::string& name, gpuStream_t stream, std::vector<PyObject*>&& objects, unsigned int flags)
{
    const EventID id = generate_unique_id();

    KernelEvent event;
    event.id = id;
    // FIXME use move for the name, and disable in non-debug builds
    event.name = name;
    event.device = get_current_device();
    event.stream = stream;
    event.used_objects = objects;

    gpuEventCreateWithFlags(&event.native_event, flags);
    gpuEventRecord(event.native_event, stream);

    event.on_start();

    UniqueLock lock(mutex);
    events.emplace(id, std::move(event));

    if (DebugSystem::is_system_enabled())
    {
        DebugSystem& debug_system = DebugSystem::get();
        debug_system.start_tracking_event(id);

        // FIXME should schedule the callback inside the debug system
        auto callback = [id]() mutable
        {
            DebugSystem& ds = DebugSystem::get();
            ds.internal_mark_event_finished(id);
        };

        gpu_host_callback(stream, callback);
    }

    return id;
}

template<bool multithreaded>
void EventManager<multithreaded>::process_finished_events()
{
    const auto has_finished_visitor = [](auto &event)
    {
        return event.has_finished();
    };

    UniqueLock lock(mutex);

    size_t expired = 0;
    // Iterate through all events and expire (destroy) ones that have finished
    auto it = events.begin();
    while (it != events.end())
    {
        //const EventID event = it->first;

        if (std::visit(has_finished_visitor, it->second))
        {
            it = events.erase(it);
            expired++;
        }
        else
        {
            ++it;
        }
    }

    printf("Cleared %zu finished events\n", expired);
    fflush(stdout);
}

template<bool multithreaded>
bool EventManager<multithreaded>::has_event_finished(EventID id) const
{
    SharedLock lock(mutex);

    auto it = events.find(id);
    if (it == events.end())
    {
        // Event that has been created but is no longer stored is treated as finished
        return id <= next_event_id ? true : false;
    }

    return std::visit([](auto &event) {
        return event.has_finished();
    }, it->second);
}

template<bool multithreaded>
inline void EventManager<multithreaded>::mark_event_finished(EventID id)
{
    if (!is_valid_event(id))
    {
        return;
    }

    UniqueLock lock(mutex);

    std::visit(
        [](auto &event) { event.mark_as_finished(); },
        events.at(id)
    );
}

template<bool multithreaded>
inline EventType EventManager<multithreaded>::get_event_type(EventID id) const
{
    SharedLock lock(mutex);

    // TODO, for now its always a kernel event unless the ID is invalid
    if (!is_valid_event(id))
    {
        return EventType::eUnknown;
    }
    else
    {
        return EventType::eKernel;
    }
}

template<bool multithreaded>
inline bool EventManager<multithreaded>::is_valid_event(EventID id) const
{
    SharedLock lock(mutex);
    return map_contains(events, id);
}

template<bool multithreaded>
inline std::vector<PyObject*> EventManager<multithreaded>::get_event_pyobjects(EventID id)
{
    SharedLock lock(mutex);

    auto it = events.find(id);
    if (it == events.end())
    {
        return std::vector<PyObject*>();
    }

    // FIXME let's not use variant. It's messy
    try
    {
        return std::get<KernelEvent>(it->second).used_objects;
    }
    catch (const std::bad_variant_access& ex)
    {
        gpaw_set_runtime_error(ex.what());
        assert(false);
        return std::vector<PyObject*>();
    }

}

extern EventManager<false> g_event_manager;
extern EventManager<true> g_multithreaded_manager;

} // namespace gpaw
