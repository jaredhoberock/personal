#include <iostream>
#include <algorithm>
#include "thread_group.hpp"
#include "async.hpp"
#include "../time_invocation/time_invocation.hpp"
#include <cassert>

inline void fill(int *x, int val, std::size_t n)
{
  using namespace test::this_thread_group;

  const int gid         = get_id();
  // XXX gcc miscompiles this or something
  //const int num_threads = size();
  const int num_threads = __singleton->size();
  const int tid         = test::this_thread::get_id();

  const int i = gid * num_threads + tid;

  if(i < n)
  {
    x[i] = val;
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
  int val = 13;
  std::vector<int> ref(n), x(n);

  std::fill(ref.begin(), ref.end(), val);

  std::fill(x.begin(), x.end(), 0);
  time_invocation(1, serial_loop_fill, x.data(), val, n);
  assert(ref == x);

  double serial_time = time_invocation(1000, serial_loop_fill, x.data(), val, n);

  std::cout << "serial_loop_fill mean duration: " << serial_time << std::endl;

  std::fill(x.begin(), x.end(), 0);
  time_invocation(1, async_fill_functor, x.data(), val, n);
  assert(ref == x);

  double async_time = time_invocation(1000, serial_loop_fill, x.data(), val, n);

  std::cout << "async_fill_functor mean duration:  " << async_time << std::endl;

  std::cout << "async penalty: " << async_time / serial_time << std::endl;

  return 0;
}

