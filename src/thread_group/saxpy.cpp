#include <iostream>
#include "thread_group.hpp"
#include "async.hpp"

void saxpy(float a, float *x, float *y, std::size_t n)
{
  int i = test::this_thread_group::get_id() * test::this_thread_group::size() + test::this_thread::get_id();

  if(i < n)
  {
    x[i] = a * x[i] + y[i];
  }
}

void my_saxpy(float a, float *x, float *y, std::size_t n)
{
  test::async(n, saxpy, a, x, y, n);
}

void simple_saxpy(float a, float *x, float *y, std::size_t n)
{
  for(int i = 0; i < n; ++i)
  {
    x[i] = a * x[i] + y[i];
  }
}

int main()
{
  std::size_t n = 1 << 20;
  auto a = 1.0f;
  std::vector<float> x(n), y(n);

  my_saxpy(a, x.data(), y.data(), n);

  return 0;
}

