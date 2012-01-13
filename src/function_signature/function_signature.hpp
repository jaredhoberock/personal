#pragma once

#include <utility>

template<typename... T> struct type_list {};

// in general, assume a functor
template<typename Function>
  struct function_signature
    : function_signature<
        decltype(&Function::operator())
      >
{};

// match function pointers
template<typename Result, typename... Args>
  struct function_signature<Result(*)(Args...)>
{
  // just use a type list for now
  typedef type_list<Result,Args...> type;
};

// match member functions
template<typename Result, typename Class, typename... Args>
  struct function_signature<Result(Class::*)(Args...)>
{
  // just use a type list for now
  typedef type_list<Result,Args...> type;
};

namespace detail
{
namespace function_signature_detail
{

struct bar {};

struct test1
{
  int operator()(bar);
};

//static_assert(std::is_same<function_signature<test1>, type_list<int,bar>>::value, "problem with test1");

} // end function_signature_detail
} // end detail

