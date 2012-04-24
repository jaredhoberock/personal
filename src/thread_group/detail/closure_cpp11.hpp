#pragma once

#include <utility>
#include <type_traits>
#include <tuple>
#include "decay_copy.hpp"
#include "apply_from_tuple_cpp11.hpp"

namespace test
{
namespace detail
{


template<typename Function, typename... Args>
  struct closure
{
  closure(const Function &f, const Args&... args)
    : f(f),
      args(args...)
  {}

  closure(Function &&f, Args&&... args)
    : f(std::forward<Function>(f)),
      args(std::forward<Args>(args)...)
  {}

  void operator()()
  {
    apply_from_tuple(f, args);
  }

  Function f;
  std::tuple<Args...> args;
};

template<typename Function, typename... Args>
  closure<
    typename std::decay<Function>::type,
    typename std::decay<Args>::type...
  >
    make_closure(Function &&f, Args&&... args)
{
  typedef closure<
    typename std::decay<Function>::type,
    typename std::decay<Args>::type...
  > result_type;

  return result_type(std::forward<Function>(f),std::forward<Args>(args)...);
}


} // end detail
} // end test

