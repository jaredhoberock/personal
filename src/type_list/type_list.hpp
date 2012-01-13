#pragma once

#if defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#include "type_list_cpp11.hpp"
#else
#include "type_list_cpp03.hpp"
#endif

