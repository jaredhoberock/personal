#pragma once

#include <utility>

namespace test
{
namespace detail
{

template<typename T>
  typename std::decay<T>::type
    decay_copy(T&& v)
{
  return std::forward<T>(v);
}

} // end detail
} // end test

