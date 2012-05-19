#pragma once

#include <tr1/tuple>

namespace test
{
namespace detail
{

template<typename Function, typename Arg1>
  void apply_from_tuple(Function f, const std::tr1::tuple<Arg1> &args)
{
  f(std::tr1::get<0>(args));
}

template<typename Function, typename Arg1, typename Arg2>
  void apply_from_tuple(Function f, const std::tr1::tuple<Arg1,Arg2> &args)
{
  f(std::tr1::get<0>(args),
    std::tr1::get<1>(args));
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  void apply_from_tuple(Function f, const std::tr1::tuple<Arg1,Arg2,Arg3> &args)
{
  f(std::tr1::get<0>(args),
    std::tr1::get<1>(args),
    std::tr1::get<2>(args));
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  void apply_from_tuple(Function f, const std::tr1::tuple<Arg1,Arg2,Arg3,Arg4> &args)
{
  f(std::tr1::get<0>(args),
    std::tr1::get<1>(args),
    std::tr1::get<2>(args),
    std::tr1::get<3>(args));
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  void apply_from_tuple(Function f, const std::tr1::tuple<Arg1,Arg2,Arg3,Arg4,Arg5> &args)
{
  f(std::tr1::get<0>(args),
    std::tr1::get<1>(args),
    std::tr1::get<2>(args),
    std::tr1::get<3>(args),
    std::tr1::get<4>(args));
}

} // end detail
} // end test

