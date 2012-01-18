#pragma once

#include <cstddef>
#include <cstdio>
#include "launch_core_access.hpp"
#include "../this_thread_group.hpp"

namespace detail
{


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  __global__ void device_marshal(Function f,
                                 Arg1 arg1, std::size_t shared_offset1,
                                 Arg2 arg2, std::size_t shared_offset2,
                                 Arg3 arg3, std::size_t shared_offset3,
                                 Arg4 arg4, std::size_t shared_offset4)
{
  // initialize the runtime
  this_thread_group::detail::set_id(threadIdx.x);

  // execute the kernel
  launch_core_access::device_marshal(f,
                                     arg1, shared_offset1,
                                     arg2, shared_offset2,
                                     arg3, shared_offset3,
                                     arg4, shared_offset4);
} // end device_marshal()


} // end detail

