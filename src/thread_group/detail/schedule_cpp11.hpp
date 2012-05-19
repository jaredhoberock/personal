#pragma once

#if defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#else
#error "This file requires compiler support for c++11"
#endif

#include <utility>
#include <cstddef>
#include <thread>
#include <algorithm>
#include <iostream>
#include <thread>

namespace test
{
namespace detail
{

template<typename Function, typename... Args>
  std::pair<std::size_t, std::size_t>
    schedule_async(std::size_t num_tasks, Function&&, Args&&...)
{
  // we assume there's no need for synchronization so schedule
  // thread per group
  // XXX for CUDA, we'd first try to achieve occupancy
  return std::make_pair(num_tasks, 1u);
}

}
} 

