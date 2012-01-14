#pragma once

#if defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#include "launch_cpp11.hpp"
#else
#include "launch_cpp03.hpp"
#endif

