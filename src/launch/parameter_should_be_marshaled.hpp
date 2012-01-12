#include <type_traits>
#include <utility>

template<typename T> struct marshal_me
{
  marshal_me() = delete;
  marshal_me(const marshal_me &) = delete;
};

namespace detail
{

// a series of integers
template<unsigned int... Integers> struct integer_series {};

template<unsigned int Next, typename> struct concat_integer;

template<unsigned int Next, unsigned int... Integers>
  struct concat_integer<Next, integer_series<Integers...>>
{
  typedef integer_series<Integers..., Next> type;
};

template<unsigned int Size> struct make_integer_series;

template<> struct make_integer_series<0> { typedef integer_series<> type; };

template<unsigned int Size>
  struct make_integer_series
{
  typedef typename concat_integer<
    Size-1,
    typename make_integer_series<Size-1>::type
  >::type type;
};

template<unsigned int i, typename Function, typename IntegerSeries>
  class parameter_needs_marshaling_impl;

template<unsigned int i, typename Function, unsigned int... Integers>
  class parameter_needs_marshaling_impl<i,Function,integer_series<Integers...>>
{
  typedef char                      yes_type;
  typedef struct { char array[2]; } no_type;

  struct only_converts_to_marshal_me
  {
    template<typename T> operator marshal_me<T> & () const;
    template<typename T> operator const marshal_me<T> & () const;

    only_converts_to_marshal_me() = delete;
    only_converts_to_marshal_me(const only_converts_to_marshal_me &) = delete;
  };
  
  struct converts_to_anything
  {
    template<typename T> operator T & () const;
    template<typename T> operator const T & () const;
  };

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
          typename detail::make_integer_series<sizeof...(Args)>::type
        >::value
      >
{};

