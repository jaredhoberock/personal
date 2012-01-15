#pragma once

#if defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#include "detail/launch_cpp11.hpp"
#else
#include "detail/launch_cpp03.hpp"
#endif

