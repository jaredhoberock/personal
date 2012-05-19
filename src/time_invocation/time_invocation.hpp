#pragma once

#if defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#include "time_invocation_cpp11.hpp"
#else
#include "time_invocation_cpp03.hpp"
#endif

