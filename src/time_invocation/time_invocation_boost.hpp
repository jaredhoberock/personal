#pragma once

#include <cstddef>
#include <boost/chrono.hpp>

template<typename Function, typename Arg1>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1)
{
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1);
  }
  boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();

  boost::chrono::milliseconds msecs = end - start;

  // return mean msecs
  return double(msecs.count()) / num_trials;
}

template<typename Function, typename Arg1, typename Arg2>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2)
{
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2);
  }
  boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();

  boost::chrono::milliseconds msecs = end - start;

  // return mean msecs
  return double(msecs.count()) / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3);
  }
  boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();

  boost::chrono::milliseconds msecs = end - start;

  // return mean msecs
  return double(msecs.count()) / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4);
  }
  boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();

  boost::chrono::milliseconds msecs = end - start;

  // return mean msecs
  return double(msecs.count()) / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5);
  }
  boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();

  boost::chrono::milliseconds msecs = end - start;

  // return mean msecs
  return double(msecs.count()) / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6);
  }
  boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();

  boost::chrono::milliseconds msecs = end - start;

  // return mean msecs
  return double(msecs.count()) / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
  }
  boost::chrono::system_clock::time_point end = boost::chrono::system_clock::now();

  boost::chrono::milliseconds msecs = end - start;

  // return mean msecs
  return double(msecs.count()) / num_trials;
}

