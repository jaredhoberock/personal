#pragma once

#include <utility>
#include <type_traits>
#include <tuple>

namespace test
{
namespace detail
{

namespace closure_detail
{

template<std::size_t...> struct index_pack {};

template<std::size_t N, std::size_t...S>
  struct increasing_index_pack
    : increasing_index_pack<N-1, N-1, S...>
{};

template<std::size_t...S>
struct increasing_index_pack<0, S...> {
  typedef index_pack<S...> type;
};

template<std::size_t N>
  typename increasing_index_pack<N>::type
    make_index_pack()
{
  return typename increasing_index_pack<N>::type();
}

template<typename Function, typename... Args, std::size_t... Indices>
  void apply_from_tuple(Function f, const std::tuple<Args...> &args, index_pack<Indices...>)
{
  f(std::get<Indices>(args)...);
}

template<typename Function, typename... Args>
  void apply_from_tuple(Function f, const std::tuple<Args...> &args)
{
  apply_from_tuple(f, args, make_index_pack<sizeof...(Args)>());
}

} // end closure_detail


// XXX closure is a copy of the function & args
//     we'd probably also want a reference closure
template<typename Function, typename... Args>
  struct closure
{
  closure(Function f, Args... args)
    : f(f), args(args...)
  {}

  void operator()()
  {
    closure_detail::apply_from_tuple(f, args);
  }

  Function f;
  std::tuple<Args...> args;
};

template<typename Function, typename... Args>
  closure<Function,Args...> make_closure(Function &&f, Args&&... args)
{
  return closure<Function,Args...>(std::forward<Function>(f),std::forward<Args>(args)...);
}

template<typename Function, typename... Args>
  closure<Function&&,Args&&...> forward_as_closure(Function &&f, Args&&... args)
{
  return closure<Function&&,Args&&...>(std::forward<Function>(f),std::forward<Args>(args)...);
}


} // end detail
} // end test

