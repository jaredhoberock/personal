#pragma once

#include <cstddef>

namespace test
{
namespace detail
{


template<typename ThreadGroup, typename Function, typename Tuple>
  class thread_group_serializer
{
  public:
    Function f;
    std::size_t num_threads;
    Tuple args;

    thread_group_serializer(std::size_t num_threads, Function f, Tuple args)
      : num_threads(num_threads),
        f(f),
        args(args)
    {}

    inline void operator()(std::size_t first_group_id, std::size_t last_group_id) const
    {
      // serially instantiate groups
      for(; first_group_id != last_group_id; ++first_group_id)
      {
        ThreadGroup(first_group_id,num_threads,f,args);
      }
    }
};

template<typename ThreadGroup, typename Function, typename Tuple>
thread_group_serializer<ThreadGroup,Function,Tuple> make_thread_group_serializer(std::size_t num_threads, Function f, Tuple args)
{
  return thread_group_serializer<ThreadGroup,Function,Tuple>(num_threads, f, args);
}


}
}

