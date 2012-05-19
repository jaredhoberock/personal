#pragma once

#include <cstddef>
#include <tbb/task_group.h>
#include "../thread_group.hpp"
#include "schedule_cpp11.hpp"
#include "closure_cpp11.hpp"
#include "parallel_for_async_cpp11.hpp"

#if !defined(__GNUC__) || !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This file requires compiler support for c++11"
#endif

namespace test
{
namespace detail
{

template<typename Function, typename... Args>
  void execute_thread_group(std::size_t group_id, std::size_t num_threads, Function &&f, Args&&... args)
{
  test::detail::ucontext_thread_group(group_id,num_threads,std::forward<Function>(f),std::forward<Args>(args)...);
}

// XXX g++ only compiles this correctly for 0-argument functions
template<typename Function, typename... Args>
  struct execute_thread_group_functor
{
  std::size_t group_id;
  std::size_t num_threads;
  Function f;
  std::tuple<Args...> args;

  execute_thread_group_functor(std::size_t group_id, std::size_t num_threads, Function &&f, Args&&... args)
    : group_id(group_id),
      num_threads(num_threads),
      f(f),
      args(args...)
  {}

  void operator()()
  {
    execute_thread_group(group_id,num_threads,std::forward<Function>(f));
  }
};

template<typename Function, typename... Args>
  execute_thread_group_functor<Function,Args...> make_thread_group(std::size_t group_id, std::size_t num_threads, Function &&f, Args&&... args)
{
  return execute_thread_group_functor<Function,Args...>(group_id,num_threads,std::forward<Function>(f),std::forward<Args>(args)...);
}

// XXX one could also imagine a hierarchical implementation
// XXX investigate how to implement this with a single task + parallel_for
template<typename Function, typename... Args>
  void linear_async(std::size_t num_groups, std::size_t num_threads, Function&& f, Args&&... args)
{
  tbb::task_group g;

  for(std::size_t i = 0; i < num_groups; ++i)
  {
    // XXX it would be nice to be able to forward here (instead of make_closure), but it makes execute_thread_group a mess
    //     g++-4.6 doesn't correctly capture && variables, else we could use a lambda
    //     if std::reference_wrapper could bind to lvalues, we could use std::ref for the particular
    //     arguments we'd care to forward (i.e., f & args)
    g.run(detail::make_thread_group(i,num_threads,detail::make_closure(std::forward<Function>(f),std::forward<Args>(args)...)));
    //g.run(detail::make_thread_group(i,num_threads,std::forward<Function>(f),std::forward<Args>(args)...));
  } // end for i

  g.wait();
}

template<typename T, typename Enable = void>
  struct is_signature_impl
    : std::false_type
{};

template<typename Function, typename... Args>
  struct is_signature_impl<
    Function(Args...),
    decltype(std::declval<Function>()(std::declval<Args>()...))
  >
    : std::true_type
{};

template<typename T>
  struct is_signature
    : is_signature_impl<T>
{};

template<typename T, typename Result = void>
  struct enable_if_signature
    : std::enable_if<
        is_signature<T>::value,
        Result
      >
{};

} // end detail


template<typename Function, typename... Args>
  typename detail::enable_if_signature<
    Function(Args...)
  >::type
    async(std::size_t num_groups, std::size_t num_threads, Function&& f, Args&&... args)
{
  return detail::parallel_for_async(num_groups, num_threads, std::forward<Function>(f), std::forward<Args>(args)...);
} // end async()


template<typename Function, typename... Args>
  typename detail::enable_if_signature<
    Function(Args...)
  >::type
    async(std::size_t num_threads, Function&& f, Args&&... args)
{
  auto schedule = detail::schedule_async(num_threads, std::forward<Function>(f), std::forward<Args>(args)...);
  return detail::parallel_for_async(schedule.first, schedule.second, std::forward<Function>(f), std::forward<Args>(args)...);
}


} // end test

