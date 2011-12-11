#pragma once

#include <cmath>
#include <functional>

// see http://en.wikipedia.org/wiki/Associativity#Definition

template<typename ForwardIterator, typename BinaryFunction, typename BinaryPredicate>
  bool is_associative(ForwardIterator first, ForwardIterator last, BinaryFunction op, BinaryPredicate equals)
{
  for(ForwardIterator x = first; x != last; ++x)
  {
    for(ForwardIterator y = first; y != last; ++y)
    {
      for(ForwardIterator z = first; z != last; ++z)
      {
        if(!equals(op(op(*x,*y), *z), op(*x, op(*y,*z)))) return false;
      }
    }
  }

  return true;
}

template<typename ForwardIterator, typename BinaryFunction>
  bool is_associative(ForwardIterator first, ForwardIterator last, BinaryFunction op)
{
  return is_associative(first, last, op, std::equal_to<typename BinaryFunction::result_type>());
}

