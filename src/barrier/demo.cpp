#include <iostream>
#include "jmpcta.hpp"
#include "ucontext_cta.hpp"

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


void foo()
{
  std::cout << "hello, world from thread " << this_thread::get_id() << std::endl;

  this_thread_group::barrier();

  std::cout << "after barrier in thread " << this_thread::get_id() << std::endl;
}

int main()
{
  launch(10, foo);

  std::cout << "main(): back from launch" << std::endl;

  return 0;
}

