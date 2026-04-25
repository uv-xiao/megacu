#include <cassert>

#include <megacu/program.h>
#include <megacu/views.h>

#include "examples/cuda_nvshmem/gemm_allreduce/common/gemm_allreduce.h"

int main() {
  megacu::program_builder phased;
  gemm_allreduce_phased_program::describe(phased);
  auto phased_ir = phased.ir();

  assert(phased_ir.extents.size() == 3);
  assert(phased_ir.domains.size() == 2);
  assert(phased_ir.participants.size() == 2);
  assert(phased_ir.resources.size() == 2);
  assert(phased_ir.events.size() == 1);
  assert(phased_ir.submissions.size() == 2);
  assert(phased_ir.submissions[0].op_name == ops::gemm_tile_produce::name);
  assert(phased_ir.submissions[1].op_name == ops::allreduce_tile_consume::name);
  assert(phased_ir.submissions[0].args[0].resource_slot == 0);
  assert(phased_ir.submissions[0].args[0].member_name == "a");
  assert(phased_ir.submissions[1].args[1].resource_slot == 0);
  assert(phased_ir.submissions[1].args[1].member_name == "c");

  megacu::program_builder overlap;
  gemm_allreduce_overlap_program::describe(overlap);
  auto overlap_ir = overlap.ir();

  assert(overlap_ir.extents.size() == 3);
  assert(overlap_ir.events.size() == 1);
  assert(overlap_ir.submissions.size() == 2);
  assert(overlap_ir.submissions[1].events.size() == 1);
  assert(overlap_ir.submissions[1].events[0].wait_mode ==
         megacu::event_wait_mode::blocking_device_wait);

  static_assert(sizeof(megacu::status_code) == 1);
  static_assert(sizeof(megacu::backend_id) == 2);

  return 0;
}
