#pragma once

#include <cstddef>
#include "schedule_cpp03.hpp"
#include "parallel_for_async_cpp03.hpp"

namespace test
{

//template<typename Function, typename... Args>
//  typename detail::enable_if_signature<
//    Function(Args...)
//  >::type
//    async(std::size_t num_groups, std::size_t num_threads, Function&& f, Args&&... args)
//{
//  return detail::parallel_for_async(num_groups, num_threads, std::forward<Function>(f), std::forward<Args>(args)...);
//} // end async()

template<typename Function>
  void async(std::size_t num_threads, Function f)
{
  std::tr1::tuple<> args;
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return detail::parallel_for_async(schedule.first, schedule.second, f, args);
}

template<typename Function, typename Arg1>
  void async(std::size_t num_threads, Function f, Arg1 arg1)
{
  std::tr1::tuple<Arg1> args(arg1);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return detail::parallel_for_async(schedule.first, schedule.second, f, args);
}

template<typename Function, typename Arg1, typename Arg2>
  void async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2)
{
  std::tr1::tuple<Arg1,Arg2> args(arg1,arg2);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return detail::parallel_for_async(schedule.first, schedule.second, f, args);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  void async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  std::tr1::tuple<Arg1,Arg2,Arg3> args(arg1,arg2,arg3);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return detail::parallel_for_async(schedule.first, schedule.second, f, args);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  std::tr1::tuple<Arg1,Arg2,Arg3,Arg4> args(arg1,arg2,arg3,arg4);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return detail::parallel_for_async(schedule.first, schedule.second, f, args);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  void async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  std::tr1::tuple<Arg1,Arg2,Arg3,Arg4,Arg5> args(arg1,arg2,arg3,arg4,arg5);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return detail::parallel_for_async(schedule.first, schedule.second, f, args);
}


} // end test

