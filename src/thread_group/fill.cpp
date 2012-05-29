#include <iostream>
#include <algorithm>
#include "thread_group.hpp"
#include "async.hpp"
#include "../time_invocation/time_invocation.hpp"

inline void fill(int *x, int val, std::size_t n)
{
  int i = test::this_thread_group::get_id() * test::this_thread_group::size() + test::this_thread::get_id();

  if(i < n)
  {
    *x = val;
  }
}

void async_fill_fcn_ptr(int *x, int val, std::size_t n)
{
  test::async(n, fill, x, val, n);
}

struct fill_group_task_functor
{
  inline void operator()(int *x, int val, std::size_t n)
  {
    fill(x,val,n);
  }
};

void async_fill_functor(int *x, int val, std::size_t n)
{
  test::async(n, fill_group_task_functor(), x, val, n);
}

void serial_loop_fill(int *x, int val, std::size_t n)
{
  int *x_last = x + n;
  for(; x != x_last; ++x)
  {
    *x = val;
  }
}

void serial_std_fill(int *x, int val, std::size_t n)
{
  std::fill(x, x + n, val);
}

int main()
{
  std::size_t n = 1 << 20;
  int val = 0;
  std::vector<int> x(n);

  std::cout << "serial_loop_fill mean duration: " << time_invocation(1000, serial_loop_fill, x.data(), val, n) << std::endl;;
  std::cout << "serial_std_transform_saxpy mean duration: " << time_invocation(1000, serial_std_fill, x.data(), val, n) << std::endl;;
  std::cout << "async_fill_fcn_ptr mean duration:  " << time_invocation(1000, async_fill_fcn_ptr, x.data(), val, n) << std::endl;
  std::cout << "async_fill_functor mean duration:  " << time_invocation(1000, async_fill_functor, x.data(), val, n) << std::endl;

  return 0;
}

