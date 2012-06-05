#pragma once

#include <cstddef>
#include <time.h>

template<typename Function, typename Arg1>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1)
{
  timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double msecs = 1000. * double(end.tv_sec - start.tv_sec);
  msecs += double(end.tv_nsec - start.tv_nsec) / 1000000.;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2)
{
  timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double msecs = 1000. * double(end.tv_sec - start.tv_sec);
  msecs += double(end.tv_nsec - start.tv_nsec) / 1000000.;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double msecs = 1000. * double(end.tv_sec - start.tv_sec);
  msecs += double(end.tv_nsec - start.tv_nsec) / 1000000.;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double msecs = 1000. * double(end.tv_sec - start.tv_sec);
  msecs += double(end.tv_nsec - start.tv_nsec) / 1000000.;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double msecs = 1000. * double(end.tv_sec - start.tv_sec);
  msecs += double(end.tv_nsec - start.tv_nsec) / 1000000.;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double msecs = 1000. * double(end.tv_sec - start.tv_sec);
  msecs += double(end.tv_nsec - start.tv_nsec) / 1000000.;

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  double time_invocation(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  double msecs = 1000. * double(end.tv_sec - start.tv_sec);
  msecs += double(end.tv_nsec - start.tv_nsec) / 1000000.;

  // return mean msecs
  return msecs / num_trials;
}

