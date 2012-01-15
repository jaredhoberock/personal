#pragma once

// a compile-time series of integers as a set of template parameters
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

