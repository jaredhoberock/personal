#pragma once

#include <cstddef>
#include <cstdio>
#include "launch_core_access.hpp"

namespace detail
{


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  __global__ void device_marshal(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  launch_core_access::device_marshal(f,arg1,arg2,arg3,arg4);
} // end device_marshal()


} // end detail

