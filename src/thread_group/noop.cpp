#include <iostream>
#include <algorithm>
#include "thread_group.hpp"
#include "async.hpp"
#include "../time_invocation/time_invocation.hpp"

struct noop_functor
{
  inline void operator()() const
  {
  }
};

void async_noop(std::size_t n)
{
  // note that we pass a functor, rather than a function pointer
  // to allow the compiler to inline it
  test::async(n, noop_functor());
}

void serial_noop(std::size_t n)
{
  for(; n; --n)
  {
  }
}

struct noop_body
{
  template<typename Range>
  void operator()(Range rng) const
  {
    int first = rng.begin();
    int last = rng.end();
    for(; first != last; ++first)
    {
      noop_functor()();
    }
  }
};

void noop_parallel_for(std::size_t n)
{
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0,n,1), noop_body());
}

int main()
{
  std::size_t n = 1 << 20;
  std::vector<float> x(n), y(n);

//  std::cout << "serial_noop mean duration: " << time_invocation(1000, serial_noop, n) << std::endl;
//  std::cout << "async_noop mean duration:  " << time_invocation(1000, async_noop, n) << std::endl;
  std::cout << "async_noop mean duration:  " << time_invocation(1000, async_noop, n) << std::endl;

  std::cout << "noop parallel_for_each mean duration: " << time_invocation(1000, noop_parallel_for, n) << std::endl;

  return 0;
}

