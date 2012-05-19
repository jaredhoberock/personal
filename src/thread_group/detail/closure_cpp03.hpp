#pragma once

#include "apply_from_tuple_cpp03.hpp"

namespace test
{
namespace detail
{


template<typename Function, typename Tuple>
  struct closure
{
  inline explicit closure(const Function &f, const Tuple& args)
    : f(f),
      args(args)
  {}

  inline void operator()()
  {
    apply_from_tuple(f, args);
  }

  Function f;
  Tuple args;
};

template<typename Function, typename Tuple>
  inline closure<Function,Tuple> make_closure(Function f, Tuple args)
{
  return closure<Function,Tuple>(f,args);
}

template<typename Function, typename Tuple>
  inline closure<Function,Tuple>
    forward_as_closure(Function f, Tuple args)
{
  return closure<Function,Tuple>(f,args);
}


} // end detail
} // end test

