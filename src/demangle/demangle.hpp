#pragma once

// see http://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html

#include <cxxabi.h>

std::string demangle(const std::string &mangled)
{
  int status;
  char *realname = abi::__cxa_demangle(mangled.c_str(), 0, 0, &status);
  std::string result(realname);
  free(realname);

  return result;
}

