#pragma once

#include <cstddef>
#include "parameter_needs_marshaling.hpp"
#include "sum_template_parameters.hpp"

namespace detail
{

template<typename Function, typename Integers, typename... Args>
  struct sizeof_marshaled_parameters_impl;

template<typename Function, unsigned int... Integers, typename... Args>
  struct sizeof_marshaled_parameters_impl<Function, integer_series<Integers...>, Args...>
{
  static const std::size_t value = sum_template_parameters<
    (parameter_needs_marshaling<Integers,Function,Args...>::value ? sizeof(Args) : 0)...
  >::value;
};

} // end detail

template<typename Function, typename... Args>
  struct sizeof_marshaled_parameters
    : detail::sizeof_marshaled_parameters_impl<
        Function,
        typename make_integer_series<sizeof...(Args)>::type,
        Args...
      >
{};


// unit tests
namespace detail
{

struct needs_marshaling
{
  void operator()(marshal_me<int> &x) {}
};

struct doesnt_need_marshaling
{
  void operator()(int x) {}
};

struct has_template
{
  template<typename T>
  void operator()(T x) {}
};

struct bar {};

struct foo
{
  void operator()(int x, bar y, marshal_me<bar> &z) {}
};

void baz(int x, marshal_me<bar> &y, const marshal_me<int> &z) {}

static_assert(sizeof_marshaled_parameters<needs_marshaling,int>::value == sizeof(int), "error with needs_marshaling");

static_assert(sizeof_marshaled_parameters<doesnt_need_marshaling,int>::value == 0u, "error with doesnt_need_marshaling");

static_assert(sizeof_marshaled_parameters<has_template,int>::value == 0u, "error with has_template");

static_assert(sizeof_marshaled_parameters<foo,int,bar,bar>::value == sizeof(bar), "error with foo");

static_assert(sizeof_marshaled_parameters<decltype(baz),int,bar,int>::value == sizeof(bar) + sizeof(int), "error with baz");

} // end detail

