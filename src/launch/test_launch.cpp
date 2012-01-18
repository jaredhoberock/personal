#include <iostream>
#include <cstdio>
#include "launch.hpp"
#include "shared.hpp"
#include "this_thread_group.hpp"

struct bar
{
  float x, y, z;

  __host__ __device__
  bar() {}

  __host__ __device__
  bar(float x, float y, float z)
    : x(x), y(y), z(z)
  {}

  __host__ __device__
  bar(const bar &other)
    : x(other.x), y(other.y), z(other.z)
  {
    printf("thread %d inside copy constructor with this = (%f,%f,%f)\n", this_thread_group::get_thread_id(), x, y, z);
  }

  __host__ __device__
  ~bar()
  {
    printf("thread %d inside destructor\n", this_thread_group::get_thread_id());
  }
};

struct foo
{
  __host__ __device__ void operator()(double x, float y, const shared<int> &z, const shared<bar> &w)
  {
    unsigned int block_idx = this_thread_group::get_block_id();
    unsigned int thread_idx = this_thread_group::get_thread_id();

    printf("thread (%d, %d) sees bar = (%f, %f, %f)\n", block_idx, thread_idx, w.get().x, w.get().y, w.get().z);
  }
};

int main()
{
  foo functor;
  launch(3, 4, functor, 10., 13.f, 13, bar(1,2,3));
  cudaThreadSynchronize();

  return 0;
}

