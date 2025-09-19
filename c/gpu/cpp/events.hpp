#pragma once

#include "gpu_core.hpp"

#include <cstdint>
#include <string>
#include <Python.h>

namespace gpaw
{

enum class EventType
{
    eUnknown,
    eKernel
};

using EventID = uint64_t;

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

    // Manually mark the event as finished
    void mark_as_finished() { cached_has_finished = true; }

    std::vector<PyObject*> used_objects;

protected:
    mutable bool cached_has_finished = false;

private:
    // Can only be constructed via EventManager
    KernelEvent() {}

    /* Called by the EventManager when the event is created and registered.
    This event takes ownership of all `PyObject`s in the `used_objects` list,
    and releases them when the event is destroyed.
    */
    void on_start();

    friend class EventManager<true>;
    friend class EventManager<false>;
};

} // namespace gpaw
