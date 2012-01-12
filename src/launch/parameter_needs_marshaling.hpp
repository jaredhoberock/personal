#include <type_traits>
#include <utility>
#include "conversion.hpp"
#include "integer_series.hpp"

template<typename T> struct marshal_me
{
  marshal_me() = delete;
  marshal_me(const marshal_me &) = delete;
};

namespace detail
{

template<unsigned int i, typename Function, typename IntegerSeries>
  class parameter_needs_marshaling_impl;

template<unsigned int i, typename Function, unsigned int... Integers>
  class parameter_needs_marshaling_impl<i,Function,integer_series<Integers...>>
{
  typedef char                      yes_type;
  typedef struct { char array[2]; } no_type;

  struct only_converts_to_marshal_me
    : only_converts_to_template1<marshal_me>
  {};

  // returns converts_to_anything when i != this_idx
  template<int inspect_idx, int this_idx>
  static typename std::enable_if<
    inspect_idx != this_idx,
    converts_to_anything
  >::type inspect_or_not();

  // returns only_converts_to_marshal_me when i == this_idx
  template<int inspect_idx, int this_idx>
  static typename std::enable_if<
    inspect_idx == this_idx,
    only_converts_to_marshal_me
  >::type inspect_or_not();

  template<unsigned int> struct helper { typedef void *type; };

  // try to call the function using something that converts to marshal_me as argument i
  // ignore all the other arguments
  template<typename UFunction> static yes_type test
  (
    typename helper
    <
      sizeof
      (
        std::declval<UFunction>()(inspect_or_not<i,Integers>()...),
        0
      )
    >::type
  );

  template<typename> static no_type test(...);

  public:
    static const bool value = sizeof(test<Function>(0)) == 1;
};

} // end detail

template<unsigned int i, typename Function, typename... Args>
  struct parameter_needs_marshaling
    : std::integral_constant<
        bool,
        detail::parameter_needs_marshaling_impl<
          i,
          Function,
          typename make_integer_series<sizeof...(Args)>::type
        >::value
      >
{};


// unit tests
namespace detail
{
namespace parameter_needs_marshaling_detail
{

struct needs_marshaling
{
  void operator()(marshal_me<int> &x);
};

struct doesnt_need_marshaling
{
  void operator()(int x);
};

struct has_template
{
  template<typename T>
  void operator()(T x);
};

struct bar {};

struct foo
{
  void operator()(int x, bar y, marshal_me<bar> &z);
};

void baz(int x, marshal_me<bar> &y, const marshal_me<int> &z);

static_assert(parameter_needs_marshaling<0,needs_marshaling,int>::value == true, "error with needs_marshaling");

static_assert(parameter_needs_marshaling<0,doesnt_need_marshaling,int>::value == false, "error with doesnt_need_marshaling");

static_assert(parameter_needs_marshaling<0,has_template,int>::value == false, "error with has_template");

static_assert(parameter_needs_marshaling<0,foo,int,bar,bar>::value == false, "error with parm 0 of foo");
static_assert(parameter_needs_marshaling<1,foo,int,bar,bar>::value == false, "error with parm 1 of foo");
static_assert(parameter_needs_marshaling<2,foo,int,bar,bar>::value == true,  "error with parm 2 of foo");

static_assert(parameter_needs_marshaling<0,decltype(baz),int,bar,int>::value == false, "error with parm 0 of baz");
static_assert(parameter_needs_marshaling<1,decltype(baz),int,bar,int>::value == true,  "error with parm 1 of baz");
static_assert(parameter_needs_marshaling<2,decltype(baz),int,bar,int>::value == true,  "error with parm 2 of baz");

} // end parameter_needs_marshaling_detail
} // end detail

