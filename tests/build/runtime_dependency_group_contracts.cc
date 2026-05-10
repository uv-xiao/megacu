#include <cassert>

#include <megacu/runtime/execution/host_orch.h>

namespace runtime = megacu::runtime;

int main() {
  runtime::execution::host_orch::frame<8, 1, 8> frame;
  auto first = frame.submit(runtime::operator_slot{1});
  auto second = frame.submit(runtime::operator_slot{2});

  auto group = frame.dependency_group();
  assert(group.push(first));
  assert(group.push(second));

  auto grouped = frame.sync(runtime::attrs(
      runtime::scheduler::depends_on_many(group.ref()),
      runtime::event_tensor::wait(runtime::event_tensor_ref{0}, 0)));
  assert(grouped == runtime::task_ref{2});
  assert(frame.current_status().code == megacu::status_code::ok);

  auto deps = frame.deps();
  auto tasks = frame.tasks();
  assert(tasks[2].dep_count == 2);
  assert(deps[tasks[2].first_dep] == first);
  assert(deps[tasks[2].first_dep + 1] == second);

  runtime::execution::host_orch::frame<8, 1, 8> mixed;
  auto a = mixed.submit(runtime::operator_slot{1});
  auto b = mixed.submit(runtime::operator_slot{2});
  auto mixed_group = mixed.dependency_group();
  assert(mixed_group.push(a));
  auto bad = mixed.sync(runtime::attrs(
      runtime::scheduler::depends_on_many(mixed_group.ref()),
      runtime::scheduler::depends_on(b)));
  assert(bad == runtime::invalid_task_ref());
  assert(mixed.current_status().code == megacu::status_code::invalid_argument);
  assert(mixed.current_status().detail == 16);

  runtime::execution::host_orch::frame<8, 1, 8> sealed;
  auto sealed_task = sealed.submit(runtime::operator_slot{1});
  auto sealed_group = sealed.dependency_group();
  assert(sealed.seal().code == megacu::status_code::ok);
  assert(!sealed_group.push(sealed_task));
  assert(sealed.current_status().code == megacu::status_code::invalid_argument);
  assert(sealed.current_status().detail == 10);
  assert(sealed.deps().empty());

  runtime::execution::host_orch::frame<8, 1, 8> interleaved;
  auto x = interleaved.submit(runtime::operator_slot{1});
  auto y = interleaved.submit(runtime::operator_slot{2});
  auto z = interleaved.submit(runtime::operator_slot{3});
  auto interleaved_group = interleaved.dependency_group();
  assert(interleaved_group.push(x));
  auto inline_task = interleaved.submit(
      runtime::operator_slot{4},
      runtime::attrs(runtime::scheduler::depends_on(y)));
  assert(inline_task == runtime::task_ref{3});
  assert(!interleaved_group.push(z));
  assert(interleaved.current_status().code ==
         megacu::status_code::invalid_argument);
  assert(interleaved.current_status().detail == 17);
  return 0;
}
