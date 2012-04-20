#include <iostream>
#include "thread_group.hpp"
#include "async.hpp"
#include <vector>
#include <numeric>
#include <cassert>
#include <utility>
#include <thread>
#include <chrono>

template<typename Iterator, typename T>
  void my_iota(Iterator first, Iterator last, T init)
{
  for(; first != last; ++first)
  {
    *first = init;
    ++init;
  }
}

std::pair<int,int> thread_coord()
{
  using namespace test;
  return std::make_pair(this_thread_group::get_id(), this_thread::get_id());
}

template<typename T1,typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1,T2> &p)
{
  return os << "(" << p.first << ", " << p.second << ")";
}

void foo()
{
  int my_thread_group = test::this_thread_group::get_id();

  std::cout << "hello, world from thread " << thread_coord() << std::endl;

  // compute a checksum
  std::vector<int> vec(100);
  int init = thread_coord().first + thread_coord().second;
  my_iota(vec.begin(), vec.end(), init);

  int before_sum = std::accumulate(vec.begin(),vec.end(),init,std::bit_xor<int>());

  test::this_thread_group::barrier();

  std::cout << "after barrier in thread " << thread_coord() << std::endl;

  vec.resize(100);
  init = thread_coord().first + thread_coord().second;
  my_iota(vec.begin(), vec.end(), init);

  int after_sum = std::accumulate(vec.begin(),vec.end(),init,std::bit_xor<int>());

  assert(before_sum == after_sum);
  assert(my_thread_group == test::this_thread_group::get_id());
}

int main()
{
  test::async(2, 10, foo);
  return 0;
}

