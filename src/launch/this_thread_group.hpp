#pragma once

#include <cstddef>

namespace this_thread_group
{

__host__ __device__
void wait()
{
#if __CUDA_ARCH__
  __syncthreads();
#else
  // XXX figure out what we could do here
#endif
} // end wait()

namespace detail
{

__device__ void *smem_ptr()
{
  extern __shared__ int smem[];
  return reinterpret_cast<void*>(smem);
} // end smem_ptr()

__device__ std::size_t *logical_block_id()
{
  // XXX probably only need one std::size_t per CTA to implement this
  return reinterpret_cast<std::size_t*>(smem_ptr());
} // end logical_thread_idx()

__device__ std::size_t *logical_thread_id()
{
  return logical_block_id() + 1 + threadIdx.x;
} // end logical_thread_idx()

__host__ __device__ std::size_t runtime_dynamic_shared_storage_requirements(std::size_t num_threads_per_block)
{
  std::size_t result = 0;

  // reserve one std::size_t for logical_block_id
  result += sizeof(std::size_t);

  // reserve two std::size_ts per thread for logical_thread_id
  result += 2 * sizeof(std::size_t) * num_threads_per_block;

  return result;
} // end runtime_dynamic_shared_storage_requirements

__device__ void *kernel_smem()
{
  // the kernel's portion of shared memory begins right after the runtime's
  return reinterpret_cast<char*>(smem_ptr()) + runtime_dynamic_shared_storage_requirements(blockDim.x);
} // end kernel_smem()

__device__ void set_block_id(std::size_t id)
{
  *logical_block_id() = id;
  wait();
} // end set_block_id()

__device__ void set_thread_id(std::size_t id)
{
  *logical_thread_id() = id;
} // end set_id()

} // end detail


__host__ __device__ std::size_t get_block_id()
{
#if __CUDA_ARCH__
  return *detail::logical_block_id();
#else
  // XXX figure out what to do here
  //     return something huge to signal a host thread
  return static_cast<std::size_t>(-1);
#endif
} // end get_block_id()

__host__ __device__ std::size_t get_thread_id()
{
#if __CUDA_ARCH__
  return *detail::logical_thread_id();
#else
  // XXX figure out what to do here
  //     return something huge to signal a host thread
  return static_cast<std::size_t>(-1);
#endif
} // end get_id()

} // end this_thread_group

