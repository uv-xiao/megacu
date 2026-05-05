#include <cassert>
#include <cstddef>
#include <cstdint>

#include <megacu/runtime.h>
#include <megacu/views.h>

namespace {

megacu::status produce(float const *, float const *,
                       megacu::symmetric_tensor_view, std::int64_t,
                       std::int64_t, std::int64_t) {
  return {};
}

megacu::status consume(megacu::symmetric_tensor_view, float *, std::int64_t,
                       std::int64_t) {
  return {};
}

struct fake_driver {
  megacu::cuda::launch_view launch;
  megacu::nvshmem::team_view team;
};

struct launch_log {
  int produce_calls = 0;
  int consume_calls = 0;
  int megakernel_calls = 0;
};

megacu::status recorded_produce(launch_log *log) {
  ++log->produce_calls;
  return {};
}

megacu::status recorded_consume(launch_log *log) {
  ++log->consume_calls;
  return {};
}

megacu::status recorded_megakernel(launch_log *log) {
  ++log->megakernel_calls;
  return {};
}

megacu::status orchestrate(fake_driver &driver, float const *a, float const *b,
                           megacu::symmetric_tensor_view partial, float *out,
                           std::int64_t m, std::int64_t n, std::int64_t k) {
  auto phase = megacu::runtime::make_phase(
      driver, megacu::runtime::progress_model::asap);
  auto ready_events = phase.event_tensor(
      megacu::runtime::attrs(megacu::runtime::event_tensor::shape(m, n)));

  auto gemm = phase.submit(
      megacu::runtime::op("gemm_tile_produce", &produce), a, b, partial, m, n,
      k,
      megacu::runtime::attrs(megacu::runtime::dispatcher::tile_grid(m, n),
                             megacu::runtime::event_tensor::notify(
                                 ready_events)));

  auto ready = phase.sync(
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(gemm),
                             megacu::runtime::event_tensor::wait(ready_events)));

  phase.submit(
      megacu::runtime::op("allreduce_tile_consume", &consume), partial, out, m,
      n,
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(ready)));

  return phase.run();
}

megacu::status orchestrate_as_single_megakernel(fake_driver &driver,
                                                launch_log *log) {
  auto phase = megacu::runtime::make_phase(
      driver, megacu::runtime::progress_model::asap);
  auto ready_events = phase.event_tensor(
      megacu::runtime::attrs(megacu::runtime::event_tensor::shape(1, 1)));

  auto produced = phase.submit(
      megacu::runtime::op("recorded_produce", &recorded_produce), log,
      megacu::runtime::attrs(
          megacu::runtime::event_tensor::notify(ready_events)));
  auto ready = phase.sync(
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(produced),
                             megacu::runtime::event_tensor::wait(ready_events)));
  phase.submit(
      megacu::runtime::op("recorded_consume", &recorded_consume), log,
      megacu::runtime::attrs(megacu::runtime::scheduler::depends_on(ready)));

  return phase.run(
      megacu::runtime::op("recorded_megakernel", &recorded_megakernel), log);
}

} // namespace

int main() {
  fake_driver driver{.launch = {.stream = nullptr, .device_ordinal = 0},
                     .team = {.team_n_pes = 2, .cuda_device_ordinal = 0}};
  float a[1]{};
  float b[1]{};
  float out[1]{};
  float partial_data[2]{};
  megacu::symmetric_tensor_view partial{
      .buffer = {.data = partial_data,
                 .bytes = static_cast<std::int64_t>(sizeof(partial_data))},
      .type = megacu::dtype::f32};

  auto status = orchestrate(driver, a, b, partial, out, 1, 1, 1);
  assert(status.code == megacu::status_code::ok);

  launch_log log;
  auto megakernel_status = orchestrate_as_single_megakernel(driver, &log);
  assert(megakernel_status.code == megacu::status_code::ok);
  assert(log.produce_calls == 0);
  assert(log.consume_calls == 0);
  assert(log.megakernel_calls == 1);
  return 0;
}
