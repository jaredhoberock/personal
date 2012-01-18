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

  template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1, std::size_t shared_offset1,
                                        const Arg2 &arg2, std::size_t shared_offset2,
                                        const Arg3 &arg3, std::size_t shared_offset3)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1,arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2,arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3,arg3));
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

  template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1, std::size_t shared_offset1,
                                        const Arg2 &arg2, std::size_t shared_offset2,
                                        const Arg3 &arg3, std::size_t shared_offset3,
                                        const Arg4 &arg4, std::size_t shared_offset4,
                                        const Arg5 &arg5, std::size_t shared_offset5)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1,arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2,arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3,arg3),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset4,arg4),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset5,arg5));
  }

  template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1, std::size_t shared_offset1,
                                        const Arg2 &arg2, std::size_t shared_offset2,
                                        const Arg3 &arg3, std::size_t shared_offset3,
                                        const Arg4 &arg4, std::size_t shared_offset4,
                                        const Arg5 &arg5, std::size_t shared_offset5,
                                        const Arg6 &arg6, std::size_t shared_offset6)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1,arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2,arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3,arg3),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset4,arg4),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset5,arg5),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset6,arg6));
  }

  template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1, std::size_t shared_offset1,
                                        const Arg2 &arg2, std::size_t shared_offset2,
                                        const Arg3 &arg3, std::size_t shared_offset3,
                                        const Arg4 &arg4, std::size_t shared_offset4,
                                        const Arg5 &arg5, std::size_t shared_offset5,
                                        const Arg6 &arg6, std::size_t shared_offset6,
                                        const Arg7 &arg7, std::size_t shared_offset7)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1,arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2,arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3,arg3),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset4,arg4),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset5,arg5),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset6,arg6),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset7,arg7));
  }

  template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1, std::size_t shared_offset1,
                                        const Arg2 &arg2, std::size_t shared_offset2,
                                        const Arg3 &arg3, std::size_t shared_offset3,
                                        const Arg4 &arg4, std::size_t shared_offset4,
                                        const Arg5 &arg5, std::size_t shared_offset5,
                                        const Arg6 &arg6, std::size_t shared_offset6,
                                        const Arg7 &arg7, std::size_t shared_offset7,
                                        const Arg8 &arg8, std::size_t shared_offset8)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1,arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2,arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3,arg3),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset4,arg4),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset5,arg5),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset6,arg6),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset7,arg7),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset8,arg8));
  }

  template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1, std::size_t shared_offset1,
                                        const Arg2 &arg2, std::size_t shared_offset2,
                                        const Arg3 &arg3, std::size_t shared_offset3,
                                        const Arg4 &arg4, std::size_t shared_offset4,
                                        const Arg5 &arg5, std::size_t shared_offset5,
                                        const Arg6 &arg6, std::size_t shared_offset6,
                                        const Arg7 &arg7, std::size_t shared_offset7,
                                        const Arg8 &arg8, std::size_t shared_offset8,
                                        const Arg9 &arg9, std::size_t shared_offset9)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1,arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2,arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3,arg3),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset4,arg4),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset5,arg5),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset6,arg6),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset7,arg7),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset8,arg8),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset9,arg9));
  }

  template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
  __device__ static void device_marshal(Function f,
                                        const Arg1 &arg1,   std::size_t shared_offset1,
                                        const Arg2 &arg2,   std::size_t shared_offset2,
                                        const Arg3 &arg3,   std::size_t shared_offset3,
                                        const Arg4 &arg4,   std::size_t shared_offset4,
                                        const Arg5 &arg5,   std::size_t shared_offset5,
                                        const Arg6 &arg6,   std::size_t shared_offset6,
                                        const Arg7 &arg7,   std::size_t shared_offset7,
                                        const Arg8 &arg8,   std::size_t shared_offset8,
                                        const Arg9 &arg9,   std::size_t shared_offset9,
                                        const Arg10 &arg10, std::size_t shared_offset10)
  {
    extern __shared__ int smem [];

    f(detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset1, arg1),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset2, arg2),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset3, arg3),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset4, arg4),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset5, arg5),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset6, arg6),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset7, arg7),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset8, arg8),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset9, arg9),
      detail::marshal_arg(reinterpret_cast<char*>(smem) + shared_offset10,arg10));
  }
};

