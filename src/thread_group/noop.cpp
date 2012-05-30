#include <iostream>
#include <algorithm>
#include "thread_group.hpp"
#include "async.hpp"
#include "../time_invocation/time_invocation.hpp"

struct noop_functor
{
  inline void operator()()
  {
    ;
  }
};

void async_noop(std::size_t n)
{
  test::async(n, noop_functor());
}

void serial_loop_noop(std::size_t n)
{
  for(; n; --n)
  {
  }
}

int main()
{
  std::size_t n = 1 << 20;

  time_invocation(10, serial_loop_noop, n);
  time_invocation(10, async_noop, n);

  std::cout << "serial_loop_noop mean duration: " << time_invocation(1000, serial_loop_noop, n) << std::endl;;
  std::cout << "async_noop_functor mean duration:  " << time_invocation(1000, async_noop, n) << std::endl;

  return 0;
}

