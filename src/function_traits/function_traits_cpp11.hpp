#pragma once

#include <utility>
#include "../type_list/type_list.hpp"

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
  typedef type_list<Result,Args...> type;
};

// match member functions
template<typename Result, typename Class, typename... Args>
  struct function_signature<Result(Class::*)(Args...)>
{
  typedef type_list<Result,Args...> type;
};

template<typename Function>
  struct function_result
    : type_list_head<
        typename function_signature<Function>::type
      >
{};

template<typename Function>
  struct function_parameters
    : type_list_tail<
        typename function_signature<Function>::type
      >
{};

template<unsigned int i, typename Function>
  struct function_parameter
    : type_list_element<
        i,
        typename function_parameters<Function>::type
      >
{};

namespace detail
{
namespace function_traits_detail
{

struct bar {};

struct test1
{
  int operator()(bar);
};

static_assert(std::is_same<function_signature<test1>::type, type_list<int,bar>>::value, "problem with test1");


struct test2
{
  int operator()(int, bar);
};

static_assert(std::is_same<function_result<test2>::type, int>::value, "problem with test2");
static_assert(std::is_same<function_parameters<test2>::type, type_list<int,bar>>::value, "problem with test2");


struct test3
{
  int operator()();
};

static_assert(std::is_same<function_parameters<test3>::type, type_list<>>::value, "problem with test3");


struct test4
{
  void operator()(int, float, double, char);
};

static_assert(std::is_same<function_parameter<0,test4>::type, int>::value, "problem with test4");
static_assert(std::is_same<function_parameter<1,test4>::type, float>::value, "problem with test4");
static_assert(std::is_same<function_parameter<2,test4>::type, double>::value, "problem with test4");
static_assert(std::is_same<function_parameter<3,test4>::type, char>::value, "problem with test4");


} // end function_traits_detail
} // end detail

