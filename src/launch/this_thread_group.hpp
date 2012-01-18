#pragma once

#include <cstddef>

namespace this_thread_group
{

namespace detail
{

__device__ void *smem_ptr()
{
  extern __shared__ int smem[];
  return reinterpret_cast<void*>(smem);
} // end smem_ptr()

__device__ std::size_t &virtual_block_id()
{
  // XXX probably only need one std::size_t per CTA to implement this
  return *reinterpret_cast<std::size_t*>(smem_ptr());
} // end virtual_thread_idx()

__device__ std::size_t &virtual_thread_id()
{
  return *(reinterpret_cast<std::size_t*>(smem_ptr()) + 1);
} // end virtual_thread_idx()

__host__ __device__ std::size_t runtime_dynamic_shared_storage_requirements(std::size_t num_threads_per_block)
{
  // reserve two std::size_ts per thread per block
  return 2 * sizeof(std::size_t) * num_threads_per_block();
} // end runtime_dynamic_shared_storage_requirements

__device__ void *dynamic_smem_begin()
{
  // the kernel's portion of shared memory begins right after the runtime's
  return reinterpret_cast<char*>(smem_ptr()) + runtime_dynamic_shared_storage_requirements(blockDim.x);
} // end dynamic_smem_begin()

__device__ void set_block_id(std::size_t id)
{
  virtual_block_id() = id;
} // end set_block_id()

__device__ void set_thread_id(std::size_t id)
{
  virtual_thread_id() = id;
} // end set_id()

} // end detail


__host__ __device__ std::size_t get_block_id()
{
#if __CUDA_ARCH__
  return virtual_block_id();
#else
  // XXX figure out what to do here
  return 0
#endif
} // end get_block_id()

__host__ __device__ std::size_t get_thread_id()
{
#if __CUDA_ARCH__
  return virtual_thread_id();
#else
  // XXX figure out what to do here
  return 0
#endif
} // end get_id()

} // end this_thread_group

