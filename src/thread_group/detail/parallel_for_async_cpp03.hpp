#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "ucontext_thread_group_cpp03.hpp"
#include "serial_thread_group_cpp03.hpp"
#include <iostream>
#include <utility>
#include <tr1/tuple>
#include <tbb/task_group.h>
#include "closure_cpp03.hpp"

namespace test
{
namespace detail
{

namespace parallel_for_async_detail
{

template<typename ThreadGroup, typename Function, typename Tuple>
  class body
{
  public:
    Function f;

    std::size_t num_threads;
    Tuple args;

    body(std::size_t num_threads, Function f, Tuple args)
      : num_threads(num_threads),
        f(f),
        args(args)
    {}

    void operator()(const tbb::blocked_range<std::size_t> &r) const
    {
      // serially instantiate groups
      for(std::size_t group_id = r.begin();
          group_id != r.end();
          ++group_id)
      {
        ThreadGroup(group_id,num_threads,f,args);
      }
    }
};

template<typename ThreadGroup,typename Function, typename Tuple>
  body<ThreadGroup,Function,Tuple> make_body(std::size_t num_threads, Function f, Tuple args)
{
  return body<ThreadGroup,Function,Tuple>(num_threads, f, args);
}

template<typename Range, typename Body>
  struct call_parallel_for
{
  Range rng;
  Body body;

  call_parallel_for(Range rng, Body body)
    : rng(rng), body(body)
  {}

  void operator()() const
  {
    tbb::parallel_for(rng,body);
  }
};

template<typename Range, typename Body>
  call_parallel_for<Range,Body> make_call_parallel_for(Range rng, Body body)
{
  return call_parallel_for<Range,Body>(rng,body);
}

} // end parallel_for_async_detail

template<typename Function, typename Tuple>
  void parallel_for_async(std::size_t num_groups, std::size_t num_threads, Function f, Tuple args)
{
  using namespace parallel_for_async_detail;

  // XXX investigate whether it makes sense to tune this 
  const std::size_t thread_groups_per_body = 1u;

  // XXX consider deriving something from tbb::task instead -- we don't actually need a group of tasks
  tbb::task_group g;

  if(num_threads > 1)
  {
    g.run(make_call_parallel_for(tbb::blocked_range<std::size_t>(0, num_groups, thread_groups_per_body), make_body<ucontext_thread_group>(num_threads,f,args)));
  }
  else
  {
    g.run(make_call_parallel_for(tbb::blocked_range<std::size_t>(0, num_groups, thread_groups_per_body), make_body<serial_thread_group>(num_threads,f,args)));
  }

  g.wait();
}

} // end detail
} // end test

