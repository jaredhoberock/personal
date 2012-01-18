#pragma once

#include <cstddef>

#include "device_marshal.hpp"
#include "shared_storage_requirements_calculator.hpp"
#include "this_thread_group.hpp"
#include <numeric>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace detail
{


template<typename Function>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    Function f)
{
  device_marshal<<<num_blocks,num_threads_per_block>>>(f);
} // end launch()


template<typename Function, typename Arg1>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0],
                                                                              arg2,storage_requirements[1]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0],
                                                                              arg2,storage_requirements[1],
                                                                              arg3,storage_requirements[2]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  // compute storage requirements for runtime
  std::size_t num_runtime_bytes = this_thread_group::detail::runtime_dynamic_shared_storage_requirements(num_threads_per_block);
  
  // compute storage requirements for parms
  std::size_t num_parm_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  // the number of dynamically-allocated smem bytes we need
  // is the sum of the runtime's requirements and the requirements of the kernel parameters
  std::size_t num_dynamic_smem_bytes = num_runtime_bytes + num_parm_bytes;

  // exclusive scan the parameter storage requirements to find each parameter's offset
  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  // launch the runtime
  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,num_runtime_bytes + storage_requirements[0],
                                                                              arg2,num_runtime_bytes + storage_requirements[1],
                                                                              arg3,num_runtime_bytes + storage_requirements[2],
                                                                              arg4,num_runtime_bytes + storage_requirements[3]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0],
                                                                              arg2,storage_requirements[1],
                                                                              arg3,storage_requirements[2],
                                                                              arg4,storage_requirements[3],
                                                                              arg5,storage_requirements[4]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0],
                                                                              arg2,storage_requirements[1],
                                                                              arg3,storage_requirements[2],
                                                                              arg4,storage_requirements[3],
                                                                              arg5,storage_requirements[4],
                                                                              arg6,storage_requirements[5]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0],
                                                                              arg2,storage_requirements[1],
                                                                              arg3,storage_requirements[2],
                                                                              arg4,storage_requirements[3],
                                                                              arg5,storage_requirements[4],
                                                                              arg6,storage_requirements[5],
                                                                              arg7,storage_requirements[6]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0],
                                                                              arg2,storage_requirements[1],
                                                                              arg3,storage_requirements[2],
                                                                              arg4,storage_requirements[3],
                                                                              arg5,storage_requirements[4],
                                                                              arg6,storage_requirements[5],
                                                                              arg7,storage_requirements[6],
                                                                              arg8,storage_requirements[7]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1,storage_requirements[0],
                                                                              arg2,storage_requirements[1],
                                                                              arg3,storage_requirements[2],
                                                                              arg4,storage_requirements[3],
                                                                              arg5,storage_requirements[4],
                                                                              arg6,storage_requirements[5],
                                                                              arg7,storage_requirements[6],
                                                                              arg8,storage_requirements[7],
                                                                              arg9,storage_requirements[8]);
} // end launch()


template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
  void host_marshal(std::size_t num_blocks, std::size_t num_threads_per_block,
                    shared_storage_requirements_calculator::result_type storage_requirements,
                    Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10)
{
  std::size_t num_dynamic_smem_bytes = thrust::reduce(storage_requirements.begin(),storage_requirements.end());

  thrust::exclusive_scan(storage_requirements.begin(),storage_requirements.end(),storage_requirements.begin());

  device_marshal<<<num_blocks,num_threads_per_block,num_dynamic_smem_bytes>>>(f,
                                                                              arg1, storage_requirements[0],
                                                                              arg2, storage_requirements[1],
                                                                              arg3, storage_requirements[2],
                                                                              arg4, storage_requirements[3],
                                                                              arg5, storage_requirements[4],
                                                                              arg6, storage_requirements[5],
                                                                              arg7, storage_requirements[6],
                                                                              arg8, storage_requirements[7],
                                                                              arg9, storage_requirements[8],
                                                                              arg10,storage_requirements[9]);
} // end launch()


} // end detail

