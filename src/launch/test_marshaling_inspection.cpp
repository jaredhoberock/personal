#include "parameter_should_be_marshalled.hpp"
#include <iostream>
#include <typeinfo>

struct needs_marshaling
{
  void operator()(marshal_me<int> &x)
  {
  }
};

struct doesnt_need_marshaling
{
  void operator()(int x)
  {
  }
};

struct has_template
{
  template<typename T>
  void operator()(T x)
  {
  }
};

struct bar {};

struct foo
{
  void operator()(int x, bar y, marshal_me<bar> &z)
  {
  }
};

void baz(int x, marshal_me<bar> &y, const marshal_me<int> &z)
{
}

int main()
{
  static_assert(parameter_needs_marshaling<0,needs_marshaling,int>::value == true, "error with needs_marshaling");

  static_assert(parameter_needs_marshaling<0,doesnt_need_marshaling,int>::value == false, "error with doesnt_need_marshaling");

  static_assert(parameter_needs_marshaling<0,has_template,int>::value == false, "error with has_template");

  static_assert(parameter_needs_marshaling<0,foo,int,bar,bar>::value == false, "error with parm 0 of foo");
  static_assert(parameter_needs_marshaling<1,foo,int,bar,bar>::value == false, "error with parm 1 of foo");
  static_assert(parameter_needs_marshaling<2,foo,int,bar,bar>::value == true,  "error with parm 2 of foo");

  static_assert(parameter_needs_marshaling<0,decltype(baz),int,bar,int>::value == false, "error with parm 0 of baz");
  static_assert(parameter_needs_marshaling<1,decltype(baz),int,bar,int>::value == true,  "error with parm 1 of baz");
  static_assert(parameter_needs_marshaling<2,decltype(baz),int,bar,int>::value == true,  "error with parm 2 of baz");

  return 0;
}

