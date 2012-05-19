#pragma once

#if defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#include "detail/async_cpp11.hpp"
#else
#include "detail/async_cpp03.hpp"
#endif

