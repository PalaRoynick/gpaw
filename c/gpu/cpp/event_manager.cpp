#include "event_manager.hpp"
#include "debug_system.hpp"

namespace gpaw
{

EventManager& EventManager::get()
{
    static EventManager instance;
    return instance;
}

EventManager::EventManager()
    : next_event_id(0)
{
}

EventID EventManager::generate_unique_id()
{
    return next_event_id++;
}

EventID EventManager::create_kernel_exec_event(const std::string& name, gpuStream_t stream, std::vector<PyObject*>&& objects, unsigned int flags)
{
    const EventID id = generate_unique_id();

    KernelEvent event;
    event.id = id;
    // FIXME use move for the name, and disable in non-debug builds
    event.name = name;
    event.device = get_current_device();
    event.stream = stream;
    event.used_objects = objects;
    event.take_pyobject_ownership = events_take_pyobject_ownership;

    gpuEventCreateWithFlags(&event.native_event, flags);
    gpuEventRecord(event.native_event, stream);

    event.on_start();

    events.emplace(id, std::move(event));

    if (DebugSystem::is_system_enabled())
    {
        DebugSystem& debug_system = DebugSystem::get();
        debug_system.start_tracking_event(id);
        debug_system.insert_stream_delay(stream);

        // FIXME should schedule the callback inside the debug system
        auto callback = [id, &debug_system]() mutable
        {
            debug_system.internal_mark_event_finished(id);
        };

        gpu_host_callback(stream, callback);
    }

    return id;
}

void EventManager::process_finished_events()
{
    const auto has_finished_visitor = [](auto &event)
    {
        return event.has_finished();
    };

    // Iterate through all events and expire (destroy) ones that have finished
    auto it = events.begin();
    while (it != events.end())
    {
        if (std::visit(has_finished_visitor, it->second))
        {
            if (DebugSystem::is_system_enabled())
            {
                const EventID event = it->first;
                DebugSystem::get().stop_tracking_event(event);
            }

            it = events.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool EventManager::has_event_finished(EventID id) const
{
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

void EventManager::mark_event_finished(EventID id)
{
    if (!is_valid_event(id))
    {
        return;
    }

    std::visit(
        [](auto &event) { event.mark_as_finished(); },
        events.at(id)
    );
}

EventType EventManager::get_event_type(EventID id) const
{
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

bool EventManager::is_valid_event(EventID id) const
{
    return map_contains(events, id);
}

std::vector<PyObject*> EventManager::get_event_pyobjects(EventID id)
{
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

void EventManager::toggle_events_take_pyobject_ownership(bool enabled)
{
    events_take_pyobject_ownership = enabled;
    if (!enabled)
    {
        printf("!!! WARNING: EventManager set to NOT take PyObject ownership. This is intended for debugging only\n");
        fflush(stdout);
    }
}

std::unordered_map<std::string, std::string> EventManager::as_string_dict(EventID id) const
{
    std::unordered_map<std::string, std::string> dict;

    auto it = events.find(id);
    if (it == events.end())
    {
        // event not found
        return dict;
    }

    // FIXME don't use variant
    try
    {
        const KernelEvent& event = std::get<KernelEvent>(it->second);

        dict["EventID"] = std::to_string(event.id);
        // FIXME base this on EventType
        dict["type"] = "GPU Kernel";
        dict["finished"] = event.has_finished() ? "true" : "false";
        dict["stream"] = stream_to_string(event.stream);
        dict["debug_name"] = event.name;

        return dict;
    }
    catch (const std::bad_variant_access& ex)
    {
        gpaw_set_runtime_error(ex.what());
        assert(false);
        return std::unordered_map<std::string, std::string>();
    }

}

} // namespace gpaw
