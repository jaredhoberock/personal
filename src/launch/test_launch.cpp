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
  bar(const bar &other)
    : x(other.x), y(other.y), z(other.z)
  {
#if __CUDA_ARCH__
    printf("thread %d inside copy constructor\n", threadIdx.x);
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
  }
};

int main()
{
  launch(7, 13, foo(), 10., 13.f, 13, bar());

  std::cout << "shared size should be " << sizeof(int) + sizeof(bar) << std::endl;

  return 0;
}

