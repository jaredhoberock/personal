#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "ucontext_thread_group_cpp11.hpp"
#include "serial_thread_group_cpp11.hpp"
#include <iostream>
#include <utility>
#include <tuple>
#include <tbb/task_group.h>
#include "closure_cpp11.hpp"

namespace test
{
namespace detail
{

namespace parallel_for_async_detail
{

// XXX g++ only compiles this correctly for 0-argument functions
template<typename ThreadGroup, typename Function, typename... Args>
  class body
{
  public:
    Function f;

    std::size_t num_threads;
    std::tuple<Args...> args;

    body(std::size_t num_threads, Function &&f, Args&&... args)
      : num_threads(num_threads),
        f(f),
        args(args...)
    {}

    void operator()(const tbb::blocked_range<std::size_t> &r) const
    {
      // serially instantiate groups
      for(auto group_id = r.begin();
          group_id != r.end();
          ++group_id)
      {
        ThreadGroup(group_id,num_threads,std::forward<Function>(f));
      }
    }
};

template<typename ThreadGroup,typename Function, typename... Args>
  body<ThreadGroup,Function,Args...> make_body(std::size_t num_threads, Function &&f, Args&&... args)
{
  return body<ThreadGroup,Function,Args...>(num_threads, std::forward<Function>(f), std::forward<Args>(args)...);
}

} // end parallel_for_async_detail

template<typename Function, typename... Args>
  void parallel_for_async(std::size_t num_groups, std::size_t num_threads, Function&& f, Args&&... args)
{
  using namespace parallel_for_async_detail;

  // XXX investigate whether it makes sense to tune this 
  const std::size_t thread_groups_per_body = 1u;

  auto closure = detail::make_closure(std::forward<Function>(f),std::forward<Args>(args)...);

  // XXX consider deriving something from tbb::task instead -- we don't actually need a group of tasks
  tbb::task_group g;

  // XXX g++-4.6 can't capture parameter packs it seems
  g.run([=]()
  {
    if(num_threads > 1)
    {
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, num_groups, thread_groups_per_body), make_body<ucontext_thread_group>(num_threads,closure));
    }
    else
    {
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, num_groups, thread_groups_per_body), make_body<serial_thread_group>(num_threads,closure));
    }
  });

  g.wait();
}

} // end detail
} // end test

