#pragma once

#include <cstddef>
#include <tr1/type_traits>
#include "schedule_cpp03.hpp"
#include "parallel_for_async_cpp03.hpp"
#include "inplace_async_cpp03.hpp"

namespace test
{

namespace async_detail
{

template<bool condition, typename Result = void>
  struct enable_if
{
  typedef Result type;
};

template<typename Result>
  struct enable_if<false,Result>
{};

template<typename T>
  struct disable_if_integral
    : enable_if<!std::tr1::is_integral<T>::value>
{};


template<typename Function, typename Tuple>
  void async_from_tuple(std::size_t num_groups, std::size_t num_threads, Function f, Tuple args)
{
//  return detail::parallel_for_async(num_groups, num_threads, f, args);
  return detail::inplace_async(num_groups, num_threads, f, args);
}

}


template<typename Function>
  void async(std::size_t num_groups, std::size_t num_threads, Function f)
{
  std::tr1::tuple<> args;
  return async_detail::async_from_tuple(num_groups, num_threads, f, args);
} // end async()

template<typename Function, typename Arg1>
  void async(std::size_t num_groups, std::size_t num_threads, Function f, Arg1 arg1)
{
  std::tr1::tuple<Arg1> args(arg1);
  return async_detail::async_from_tuple(num_groups, num_threads, f, args);
} // end async()

template<typename Function, typename Arg1, typename Arg2>
  void async(std::size_t num_groups, std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2)
{
  std::tr1::tuple<Arg1,Arg2> args(arg1,arg2);
  return async_detail::async_from_tuple(num_groups, num_threads, f, args);
} // end async()

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  void async(std::size_t num_groups, std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  std::tr1::tuple<Arg1,Arg2,Arg3> args(arg1,arg2,arg3);
  return async_detail::async_from_tuple(num_groups, num_threads, f, args);
} // end async()

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void async(std::size_t num_groups, std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  std::tr1::tuple<Arg1,Arg2,Arg3,Arg4> args(arg1,arg2,arg3,arg4);
  return async_detail::async_from_tuple(num_groups, num_threads, f, args);
} // end async()


template<typename Function>
  typename async_detail::disable_if_integral<Function>::type
    async(std::size_t num_threads, Function f)
{
  std::tr1::tuple<> args;
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return async(schedule.first, schedule.second, f);
}

template<typename Function, typename Arg1>
  typename async_detail::disable_if_integral<Function>::type
    async(std::size_t num_threads, Function f, Arg1 arg1)
{
  std::tr1::tuple<Arg1> args(arg1);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return async(schedule.first, schedule.second, f, arg1);
}

template<typename Function, typename Arg1, typename Arg2>
  typename async_detail::disable_if_integral<Function>::type
    async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2)
{
  std::tr1::tuple<Arg1,Arg2> args(arg1,arg2);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return async(schedule.first, schedule.second, f, arg1, arg2);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  typename async_detail::disable_if_integral<Function>::type
    async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  std::tr1::tuple<Arg1,Arg2,Arg3> args(arg1,arg2,arg3);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return async(schedule.first, schedule.second, f, arg1, arg2, arg3);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  typename async_detail::disable_if_integral<Function>::type
    async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  std::tr1::tuple<Arg1,Arg2,Arg3,Arg4> args(arg1,arg2,arg3,arg4);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return async(schedule.first, schedule.second, f, arg1, arg2, arg3, arg4);
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  typename async_detail::disable_if_integral<Function>::type
    async(std::size_t num_threads, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  std::tr1::tuple<Arg1,Arg2,Arg3,Arg4,Arg5> args(arg1,arg2,arg3,arg4,arg5);
  std::pair<std::size_t,std::size_t> schedule = detail::schedule_async(num_threads, f, args);
  return async_detail::async_from_tuple(schedule.first, schedule.second, f, args);
}


} // end test

