#pragma once

#include "thread_group_serializer_cpp03.hpp"
#include "ucontext_thread_group_cpp03.hpp"
#include "serial_thread_group_cpp03.hpp"

namespace test
{
namespace detail
{

template<typename Function, typename Tuple>
  inline void inplace_async(std::size_t num_groups, std::size_t num_threads, Function f, Tuple args)
{
  if(num_threads > 1)
  {
    make_thread_group_serializer<ucontext_thread_group>(num_threads, f, args)(0, num_groups);
  }
  else
  {
    make_thread_group_serializer<serial_thread_group>(num_threads, f, args)(0, num_groups);
  }
}

} // end detail
} // end test

