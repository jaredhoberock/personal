#pragma once

#include "marshaled_args.hpp"

struct launch_core_access
{
  template<typename T>
    static shared<T> &shared_dummy()
  {
    typename shared<T>::construct_dummy_tag tag;
    static shared<T> dummy((tag));
    return dummy;
  }

  template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1, std::size_t shared_offset1,
                                        const Arg2 &arg2, std::size_t shared_offset2,
                                        const Arg3 &arg3, std::size_t shared_offset3,
                                        const Arg4 &arg4, std::size_t shared_offset4)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1,arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2,arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3,arg3),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset4,arg4));
  }
};

