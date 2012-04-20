#pragma once

#include "thread_group.hpp"
#include <cstddef>
#include <tbb/task_group.h>

namespace test
{
namespace detail
{

template<typename Function>
  struct execute_thread_group
{
  int id;
  std::size_t num_threads;
  Function f;

  execute_thread_group(int id, std::size_t num_threads, Function f)
    : id(id),num_threads(num_threads),f(f)
  {}

  void operator()()
  {
    test::detail::ucontext_thread_group(id,num_threads,f);
  }
};

template<typename Function>
  execute_thread_group<Function> make_thread_group(std::size_t group_id, std::size_t num_threads, Function f)
{
  return execute_thread_group<Function>(group_id,num_threads,f);
}

// XXX one could also imagine a hierarchical implementation
template<typename Function>
  void linear_async(std::size_t num_groups, std::size_t num_threads, Function f)
{
  tbb::task_group g;

  for(std::size_t i = 0; i < num_groups; ++i)
  {
    g.run(detail::make_thread_group(i, num_threads,f));
  } // end for i

  g.wait();
}

} // end detail


template<typename Function>
  void async(std::size_t num_groups, std::size_t num_threads, Function f)
{
  return detail::linear_async(num_groups, num_threads, f);
} // end async()


} // end test

