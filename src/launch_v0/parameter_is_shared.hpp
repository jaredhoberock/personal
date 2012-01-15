#include <type_traits>
#include <utility>
#include "conversion.hpp"
#include "integer_series.hpp"
#include "shared.hpp"
#include "sfinae_types.hpp"

namespace detail
{

template<unsigned int i, typename Function, typename IntegerSeries>
  class parameter_is_shared_impl;

template<unsigned int i, typename Function, unsigned int... Integers>
  class parameter_is_shared_impl<i,Function,integer_series<Integers...>>
    : sfinae_types
{
  struct only_converts_to_shared
    : only_converts_to_template1<shared>
  {};

  // returns converts_to_anything when i != this_idx
  template<int inspect_idx, int this_idx>
  static typename std::enable_if<
    inspect_idx != this_idx,
    converts_to_anything
  >::type inspect_or_not();

  // returns only_converts_to_shared when i == this_idx
  template<int inspect_idx, int this_idx>
  static typename std::enable_if<
    inspect_idx == this_idx,
    only_converts_to_shared
  >::type inspect_or_not();

  template<unsigned int> struct helper
  {
    typedef void *type;
  };

  // try to call the function using something that converts to shared as argument i
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

  template<typename...> static no_type test(...);

  public:
    static const bool value = sizeof(test<Function>(0)) == 1;
};

} // end detail

template<unsigned int i, typename Function, typename... Args>
  struct parameter_is_shared
    : std::integral_constant<
        bool,
        detail::parameter_is_shared_impl<
          i,
          Function,
          typename make_integer_series<sizeof...(Args)>::type
        >::value
      >
{};


// unit tests
namespace detail
{
namespace parameter_is_shared_detail
{

struct has_shared
{
  void operator()(shared<int> &x);
};

struct doesnt_have_shared
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
  void operator()(int x, bar y, shared<bar> &z);
};

void baz(int x, shared<bar> &y, const shared<int> &z);

static_assert(parameter_is_shared<0,has_shared,int>::value == true, "error with has_shared");

static_assert(parameter_is_shared<0,doesnt_have_shared,int>::value == false, "error with doesnt_have_shared");

static_assert(parameter_is_shared<0,has_template,int>::value == false, "error with has_template");

static_assert(parameter_is_shared<0,foo,int,bar,bar>::value == false, "error with parm 0 of foo");
static_assert(parameter_is_shared<1,foo,int,bar,bar>::value == false, "error with parm 1 of foo");
static_assert(parameter_is_shared<2,foo,int,bar,bar>::value == true,  "error with parm 2 of foo");

static_assert(parameter_is_shared<0,decltype(baz),int,bar,int>::value == false, "error with parm 0 of baz");
static_assert(parameter_is_shared<1,decltype(baz),int,bar,int>::value == true,  "error with parm 1 of baz");
static_assert(parameter_is_shared<2,decltype(baz),int,bar,int>::value == true,  "error with parm 2 of baz");

} // end parameter_is_shared_detail
} // end detail

