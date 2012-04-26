#pragma once

#include <utility>
#include <type_traits>
#include <tuple>
#include <functional>
#include "apply_from_tuple_cpp11.hpp"

namespace test
{
namespace detail
{


template<typename Function, typename... Args>
  struct closure
{
  explicit closure(const Function &f, const Args&... args)
    : f(f),
      args(args...)
  {}

  template<typename OtherFunction, typename... OtherArgs>
    explicit closure(OtherFunction &&f, OtherArgs&&... args,
                     typename std::enable_if<
                       std::is_convertible<OtherFunction,Function>::value
                     >::type * = 0)
      : f(std::forward<OtherFunction>(f)),
        args(std::forward<OtherArgs>(args)...)
    {}

  void operator()()
  {
    apply_from_tuple(f, args);
  }

  Function f;
  std::tuple<Args...> args;
};

template<typename T>
  struct strip_reference_wrapper
{
  typedef T type;
};

template<typename T>
  struct strip_reference_wrapper<std::reference_wrapper<T> >
{
  typedef T& type;
};

template<typename T>
  struct decay_and_strip_reference_wrapper
    : strip_reference_wrapper<
        typename std::decay<T>::type
      >
{};

template<typename Function, typename... Args>
  closure<
    typename decay_and_strip_reference_wrapper<Function>::type,
    typename decay_and_strip_reference_wrapper<Args>::type...
  >
    make_closure(Function &&f, Args&&... args)
{
  typedef closure<
    typename decay_and_strip_reference_wrapper<Function>::type,
    typename decay_and_strip_reference_wrapper<Args>::type...
  > result_type;

  return result_type(std::forward<Function>(f),std::forward<Args>(args)...);
}

template<typename Function, typename... Args>
  closure<Function&&,Args&&...>
    forward_as_closure(Function &&f, Args&&... args)
{
  return closure<Function&&,Args&&...>(std::forward<Function>(f),std::forward<Args>(args)...);
}


} // end detail
} // end test

