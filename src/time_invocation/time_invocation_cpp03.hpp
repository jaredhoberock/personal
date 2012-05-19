#pragma once

#include <cstddef>
#include <ctime>

template<typename Function, typename Arg1>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1)
{
  std::clock_t start = std::clock();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1);
  }
  std::clock_t end = std::clock();

  double msecs = 1000.0 * double(end - start) / CLOCKS_PER_SEC;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2)
{
  std::clock_t start = std::clock();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2);
  }
  std::clock_t end = std::clock();

  double msecs = 1000.0 * double(end - start) / CLOCKS_PER_SEC;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  std::clock_t start = std::clock();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3);
  }
  std::clock_t end = std::clock();

  double msecs = 1000.0 * double(end - start) / CLOCKS_PER_SEC;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  std::clock_t start = std::clock();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4);
  }
  std::clock_t end = std::clock();

  double msecs = 1000.0 * double(end - start) / CLOCKS_PER_SEC;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  std::clock_t start = std::clock();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5);
  }
  std::clock_t end = std::clock();

  double msecs = 1000.0 * double(end - start) / CLOCKS_PER_SEC;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  std::clock_t start = std::clock();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6);
  }
  std::clock_t end = std::clock();

  double msecs = 1000.0 * double(end - start) / CLOCKS_PER_SEC;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  std::clock_t start = std::clock();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
  }
  std::clock_t end = std::clock();

  double msecs = 1000.0 * double(end - start) / CLOCKS_PER_SEC;

  // return mean msecs
  return msecs / num_trials;
}

