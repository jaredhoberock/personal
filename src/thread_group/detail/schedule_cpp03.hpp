#pragma once

#include <utility>
#include <cstddef>

namespace test
{
namespace detail
{

template<typename Function, typename Tuple>
  std::pair<std::size_t, std::size_t>
    schedule_async(std::size_t num_tasks, const Function &, const Tuple&)
{
  // we assume there's no need for synchronization so schedule
  // thread per group
  // XXX for CUDA, we'd first try to achieve occupancy
  return std::make_pair(num_tasks, 1u);
}

}
} 

