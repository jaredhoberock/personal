#pragma once

#if defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#else
#error "This file requires compiler support for c++11"
#endif

#include <cstddef>
#include <chrono>

template<typename Function, typename... Args>
  double time_invocation(std::size_t num_trials, Function &&f, Args&&... args)
{
  auto start = std::chrono::high_resolution_clock::now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(std::forward<Args&&>(args)...);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // return mean duration in msecs
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

