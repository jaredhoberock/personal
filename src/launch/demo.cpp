#include <iostream>
#include <cstdio>
#include "launch.hpp"
#include "shared.hpp"
#include "this_thread_group.hpp"

struct bar
{
  float x, y, z;

  __host__ __device__ bar() {}

  __host__ __device__ bar(float x, float y, float z)
    : x(x), y(y), z(z)
  {}

  __host__ __device__ bar(const bar &other)
    : x(other.x), y(other.y), z(other.z)
  {
    unsigned int id = this_thread_group::get_thread_id();
    printf("thread %u inside copy constructor with this = (%f,%f,%f)\n", id, x, y, z);
  }

  __host__ __device__ ~bar()
  {
    unsigned int id = this_thread_group::get_thread_id();
    printf("thread %d inside destructor\n", id);
  }
};

struct kernel
{
  __host__ __device__ void operator()(double x, float y, const shared<int> &z, const shared<bar> &w)
  {
    unsigned int block_idx = this_thread_group::get_block_id();
    unsigned int thread_idx = this_thread_group::get_thread_id();

    printf("thread (%d, %d) inside kernel::operator() sees bar = (%f, %f, %f)\n", block_idx, thread_idx, w.get().x, w.get().y, w.get().z);

    // thread 0 modifies w
    this_thread_group::wait();
    if(thread_idx == 0)
    {
      w.get().x = 3;
      w.get().y = 2;
      w.get().z = 1;
    }
    this_thread_group::wait();

    printf("after barrier, thread (%d, %d) inside kernel::operator() sees bar = (%f, %f, %f)\n", block_idx, thread_idx, w.get().x, w.get().y, w.get().z);
  }
};

int main()
{
  // launch the kernel with 3 CTAs of 4 threads each
  launch(3, 4, kernel(), 10., 13.f, 13, bar(1,2,3));

  cudaError_t err = cudaThreadSynchronize();
  std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

  return 0;
}

