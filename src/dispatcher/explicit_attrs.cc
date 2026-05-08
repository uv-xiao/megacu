#include <megacu/runtime.h>

namespace megacu::runtime::dispatcher {

status map_explicit_attrs(dispatch_state &out,
                          std::span<linked_task const> tasks,
                          std::span<linked_event_tensor const> events) {
  out = {};
  out.task_count = tasks.size();
  out.event_tensor_count = events.size();

  for (auto const &event : events) {
    for (auto attr : event.attributes.entries()) {
      if (attr.kind == attr_kind::event_tensor_shape) {
        out.sync_event_count += attr.first * attr.second;
      }
    }
  }

  for (auto const &task : tasks) {
    if (task.kind == task_kind::sync_only) {
      ++out.sync_task_count;
    }

    for (auto attr : task.attributes.entries()) {
      if (attr.kind == attr_kind::dispatch_single_tile) {
        ++out.tile_grid_attrs;
        ++out.total_tile_count;
      } else if (attr.kind == attr_kind::dispatch_tile_grid) {
        ++out.tile_grid_attrs;
        out.total_tile_count += attr.first * attr.second;
      } else if (attr.kind == attr_kind::event_notify) {
        ++out.event_notify_attrs;
      } else if (attr.kind == attr_kind::event_wait) {
        ++out.event_wait_attrs;
      } else if (attr.kind == attr_kind::event_trigger) {
        ++out.event_trigger_attrs;
      }
    }
  }

  return {};
}

} // namespace megacu::runtime::dispatcher

extern "C" int megacu_dispatcher_explicit_attrs_component() { return 1; }
