#include <iostream>
#include "jmpcta.hpp"
#include "ucontext_cta.hpp"
#include <numeric>
#include <vector>
#include <functional>
#include <cassert>

template<typename Function>
  void launch(int num_threads, Function f)
{
  ucontext_cta(num_threads, f);
}

namespace this_thread
{

int get_id()
{
  return this_thread_group::current_thread_id();
}

}

template<typename Iterator, typename T>
  void iota(Iterator first, Iterator last, T init)
{
  for(; first != last; ++first)
  {
    *first = init;
    ++init;
  }
}


void foo()
{
  std::cout << "hello, world from thread " << this_thread::get_id() << std::endl;

  // compute a checksum
  std::vector<int> vec(100);
  iota(vec.begin(), vec.end(), this_thread::get_id());

  int before_sum = std::accumulate(vec.begin(),vec.end(),0,std::bit_xor<int>());

  this_thread_group::barrier();

  std::cout << "after barrier in thread " << this_thread::get_id() << std::endl;

  vec.resize(100);
  iota(vec.begin(), vec.end(), this_thread::get_id());

  int after_sum = std::accumulate(vec.begin(),vec.end(),0,std::bit_xor<int>());

  assert(before_sum == after_sum);
}

int main()
{
  launch(10, foo);

  std::cout << "main(): back from launch" << std::endl;

  return 0;
}

