#pragma once

#include "events.hpp"

#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <variant>
#include <vector>

namespace gpaw
{

// Singleton
class EventManager
{
public:

    static EventManager& get();

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

    // Can toggle ownership off for debugging
    void toggle_events_take_pyobject_ownership(bool enabled);

    // Returns info about the event in a string format
    std::unordered_map<std::string, std::string> as_string_dict(EventID id) const;

protected:

    // Generates a new globally unique event ID
    EventID generate_unique_id();

    using EventVariant = std::variant<KernelEvent>;
    std::unordered_map<EventID, EventVariant> events;

    bool events_take_pyobject_ownership = true;

private:
    // singleton pattern
    EventManager();
    ~EventManager() {}
    EventManager& operator=(const EventManager&) = delete;
    EventManager(const EventManager&) = delete;

    // Events will be created from one thread only, so this does not need to be atomic
    EventID next_event_id;
};

} // namespace gpaw
