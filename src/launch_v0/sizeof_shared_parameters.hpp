#pragma once

#include <cstddef>
#include "parameter_is_shared.hpp"
#include "sum_template_parameters.hpp"

namespace detail
{

template<typename Function, typename Integers, typename... Args>
  struct sizeof_shared_parameters_impl;

template<typename Function, unsigned int... Integers, typename... Args>
  struct sizeof_shared_parameters_impl<Function, integer_series<Integers...>, Args...>
{
  static const std::size_t value = sum_template_parameters<
    (parameter_is_shared<Integers,Function,Args...>::value ? sizeof(Args) : 0)...
  >::value;
};

} // end detail

template<typename Function, typename... Args>
  struct sizeof_shared_parameters
    : detail::sizeof_shared_parameters_impl<
        Function,
        typename make_integer_series<sizeof...(Args)>::type,
        Args...
      >
{};


// unit tests
namespace detail
{
namespace sizeof_shared_parameters_detail
{


struct has_shared
{
  void operator()(shared<int> &x) {}
};

static_assert(sizeof_shared_parameters<has_shared,int>::value == sizeof(int), "error with has_shared");


struct doesnt_have_shared
{
  void operator()(int x) {}
};

static_assert(sizeof_shared_parameters<doesnt_have_shared,int>::value == 0u, "error with doesnt_have_shared");


struct has_template
{
  template<typename T>
  void operator()(T x) {}
};

static_assert(sizeof_shared_parameters<has_template,int>::value == 0u, "error with has_template");


struct bar {};

struct foo
{
  void operator()(int x, bar y, shared<bar> &z) {}
};

static_assert(sizeof_shared_parameters<foo,int,bar,bar>::value == sizeof(bar), "error with foo");


void baz(int x, shared<bar> &y, const shared<int> &z) {}

static_assert(sizeof_shared_parameters<decltype(baz),int,bar,int>::value == sizeof(bar) + sizeof(int), "error with baz");

struct test1_struct
{
  float x, y, z;
};

void test1(double x, float y, shared<int> &z, shared<test1_struct> &w);

static_assert(sizeof_shared_parameters<decltype(test1),double,float,int,test1_struct>::value == sizeof(int) + sizeof(test1_struct), "error with test1");


} // end sizeof_shared_parameters_detail
} // end detail


