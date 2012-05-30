#include <iostream>
#include <algorithm>
#include "thread_group.hpp"
#include "async.hpp"
#include "../time_invocation/time_invocation.hpp"

inline void copy(int *src, int *dst, std::size_t n)
{
  int i = test::this_thread_group::get_id() * test::this_thread_group::size() + test::this_thread::get_id();

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
  for(; src != src_last; ++src)
  {
    *dst = *src;
  }
}

int main()
{
  std::size_t n = 1 << 20;
  std::vector<int> src(n), dst(n);

  time_invocation(2, serial_copy, src.data(), dst.data(), n);
  time_invocation(2, async_copy_functor, src.data(), dst.data(), n);

  std::cout << "serial_copy mean duration: " << time_invocation(1000, serial_copy, src.data(), dst.data(), n) << std::endl;;
  std::cout << "async_copy_functor mean duration:  " << time_invocation(1000, async_copy_functor, src.data(), dst.data(), n) << std::endl;

  return 0;
}

