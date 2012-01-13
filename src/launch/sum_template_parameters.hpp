#pragma once

#include <cstddef>

template<std::size_t...> struct sum_template_parameters;

template<std::size_t i, std::size_t... integers>
  struct sum_template_parameters<i,integers...>
{
  static const std::size_t value = i + sum_template_parameters<integers...>::value;
};

template<>
  struct sum_template_parameters<>
{
  static const std::size_t value = 0u;
};

// unit tests follow
static_assert(sum_template_parameters<>::value == 0, "error with no parm case");
static_assert(sum_template_parameters<7>::value == 7, "error with one term case");
static_assert(sum_template_parameters<7,13,42>::value == 62, "error with three term case");

