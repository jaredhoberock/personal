#pragma once

#include <cstddef>

#include "device_marshal.hpp"

namespace detail
{


template<typename Function>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    Function f)
{
  device_marshal<<<num_blocks,num_threads_per_block>>>(f);
} // end launch()


template<typename Function, typename Arg1>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1);
} // end launch()


template<typename Function, typename Arg1, typename Arg2>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  std::cout << "about to launch kernel" << std::endl;
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3,arg4);
  std::cout << "about to cudaThreadSynchronize()" << std::endl;
  cudaThreadSynchronize();
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3,arg4,arg5);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3,arg4,arg5,arg6);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block, std::size_t num_dynamic_smem_bytes,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10)
{
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10);
} // end launch()


} // end detail

