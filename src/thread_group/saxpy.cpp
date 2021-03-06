#include <iostream>
#include <algorithm>
#include "thread_group.hpp"
#include "async.hpp"
#include "../time_invocation/time_invocation.hpp"

inline void saxpy(float a, float *x, float *y, std::size_t n)
{
  int i = test::this_thread_group::get_id() * test::this_thread_group::size() + test::this_thread::get_id();

  if(i < n)
  {
    x[i] = a * x[i] + y[i];
  }
}

void async_saxpy_fcn_ptr(float a, float *x, float *y, std::size_t n)
{
  test::async(n, saxpy, a, x, y, n);
}

struct saxpy_group_task_functor
{
  void operator()(float a, float *x, float *y, std::size_t n)
  {
    saxpy(a,x,y,n);
  }
};

void async_saxpy_functor(float a, float *x, float *y, std::size_t n)
{
  test::async(n, saxpy_group_task_functor(), a, x, y, n);
}

void serial_saxpy(float a, float *x, float *y, std::size_t n)
{
  float *x_last = x + n;
  for(; x != x_last; ++x, ++y)
  {
    *x = a * (*x) + *y;
  }
}

struct saxpy_functor
{
  float a;

  saxpy_functor(float a)
    : a(a)
  {}

  float operator()(float xi, float yi)
  {
    return a * xi + yi;
  }
};

void serial_transform_saxpy(float a, float *x, float *y, std::size_t n)
{
  std::transform(x, x + n, y, x, saxpy_functor(a));
}

int main()
{
  std::size_t n = 1 << 20;
  float a = 1.0f;
  std::vector<float> x(n), y(n);

  std::cout << "serial_saxpy mean duration: " << time_invocation(1000, serial_saxpy, a, x.data(), y.data(), n) << std::endl;;
  std::cout << "serial_transform_saxpy mean duration: " << time_invocation(1000, serial_transform_saxpy, a, x.data(), y.data(), n) << std::endl;;
  std::cout << "async_saxpy_fcn_ptr mean duration:  " << time_invocation(1000, async_saxpy_fcn_ptr, a, x.data(), y.data(), n) << std::endl;
  std::cout << "async_saxpy_functor mean duration:  " << time_invocation(1000, async_saxpy_functor, a, x.data(), y.data(), n) << std::endl;

  return 0;
}

