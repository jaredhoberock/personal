#include <iostream>
#include <cstdio>
#include "launch.hpp"
#include "shared.hpp"

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
#if __CUDA_ARCH__
    printf("thread %d inside copy constructor with this = (%f,%f,%f)\n", threadIdx.x, x, y, z);
#endif
  }

  __host__ __device__
  ~bar()
  {
#if __CUDA_ARCH__
    printf("thread %d inside destructor\n", threadIdx.x);
#endif
  }
};

struct foo
{
  __host__ __device__ void operator()(double x, float y, const shared<int> &z, const shared<bar> &w)
  {
#if __CUDA_ARCH__
    printf("thread (%d, %d) sees bar = (%f, %f, %f)\n", blockIdx.x, threadIdx.x, w.get().x, w.get().y, w.get().z);
#endif
  }
};

int main()
{
  foo functor;
  launch(3, 4, functor, 10., 13.f, 13, bar(1,2,3));
  cudaThreadSynchronize();

  return 0;
}

