#include <iostream>
#include <algorithm>
#include "thread_group.hpp"
#include "async.hpp"
#include "../time_invocation/time_invocation.hpp"
#include <cassert>

inline void copy(int *src, int *dst, std::size_t n)
{
  using namespace test;
  const int i = this_thread_group::size() * this_thread_group::get_id() + test::this_thread::get_id();

  if(i < n)
  {
    dst[i] = src[i];
  }
}

struct copy_group_task_functor
{
  inline void operator()(int *src, int *dst, std::size_t n)
  {
    copy(src,dst,n);
  }
};

void async_copy_functor(int *src, int *dst, std::size_t n)
{
  test::async(n, copy_group_task_functor(), src, dst, n);
}

void serial_copy(int *src, int *dst, std::size_t n)
{
  int *src_last = src + n;
  for(; src != src_last; ++src, ++dst)
  {
    *dst = *src;
  }
}

int main()
{
  std::size_t n = 1 << 20;
  std::vector<int> ref(n), src(n), dst(n);

  for(std::size_t i = 0; i < n; ++i)
  {
    ref[i] = i;
    src[i] = i;
  }

  std::fill(dst.begin(), dst.end(), 0);
  time_invocation(1, serial_copy, src.data(), dst.data(), n);
  assert(ref == dst);

  double serial_time = time_invocation(1000, serial_copy, src.data(), dst.data(), n);

  std::cout << "serial_copy mean duration: " << serial_time << std::endl;

  std::fill(dst.begin(), dst.end(), 0);
  time_invocation(1, async_copy_functor, src.data(), dst.data(), n);
  assert(ref == dst);

  double async_time = time_invocation(1000, async_copy_functor, src.data(), dst.data(), n);

  std::cout << "async_copy_functor mean duration:  " << async_time << std::endl;

  std::cout << "async penalty: " << async_time / serial_time << std::endl;

  return 0;
}

