#pragma once

#include "gpu_core.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <variant>
#include <vector>
#include <Python.h>

namespace gpaw
{

using EventID = uint64_t;

// forward declare to make friend declaration of templates work
template<bool multithreaded> class EventManager;

struct EventBase
{
    // Unique ID for the event
    EventID id;
    // Name used for debugging
    std::string name;
};

struct KernelEvent : public EventBase
{
    // Owns a gpuStream_t object and a list of PyObjects => disallow copy, allow moves

    ~KernelEvent();
    KernelEvent(const KernelEvent&) = delete;
    KernelEvent& operator=(const KernelEvent&) = delete;
    KernelEvent(KernelEvent&& other) noexcept;
    KernelEvent& operator=(KernelEvent&& other) noexcept;

    gpuEvent_t native_event = nullptr;
    gpuStream_t stream = nullptr;
    int32_t device = -1;
    bool has_finished() const;

    std::vector<PyObject*> used_objects;

private:
    // Can only be constructed via EventManager
    KernelEvent() {}

    /* Called by the EventManager when the event is created and registered.
    This event takes ownership of all `PyObject`s in the `used_objects` list,
    and releases them when the event is destroyed.
    */
    void on_start();

    mutable bool cached_has_finished = false;

    friend class EventManager<true>;
    friend class EventManager<false>;
};

using EventVariant = std::variant<KernelEvent>;

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
    // Mark event as ready to be destroyed
    void mark_event_expired(EventID id);

    bool is_valid_event(EventID id) const;

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
inline void EventManager<multithreaded>::mark_event_expired(EventID id)
{
    if (!is_valid_event(id))
    {
        return;
    }

    // Just destroy the event immediately
    UniqueLock lock(mutex);
    events.erase(id);
}

template<bool multithreaded>
inline bool EventManager<multithreaded>::is_valid_event(EventID id) const
{
    SharedLock lock(mutex);
    return map_contains(events, id);
}

extern EventManager<false> g_event_manager;
extern EventManager<true> g_multithreaded_manager;

} // namespace gpaw
