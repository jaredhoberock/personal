#pragma once

#include "../type_list/type_list.hpp"

template<typename Function> struct function_signature;

// specializations for function ptrs
template<typename Result>
  struct function_signature<Result(*)()>
{
  typedef type_list<Result> type;
};

template<typename Result, typename Arg1>
  struct function_signature<Result(*)(Arg1)>
{
  typedef type_list<Result,Arg1> type;
};

template<typename Result, typename Arg1, typename Arg2>
  struct function_signature<Result(*)(Arg1,Arg2)>
{
  typedef type_list<Result,Arg1,Arg2> type;
};

template<typename Result, typename Arg1, typename Arg2, typename Arg3>
  struct function_signature<Result(*)(Arg1,Arg2,Arg3)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3> type;
};

template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  struct function_signature<Result(*)(Arg1,Arg2,Arg3,Arg4)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4> type;
};

template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  struct function_signature<Result(*)(Arg1,Arg2,Arg3,Arg4,Arg5)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5> type;
};

template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  struct function_signature<Result(*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6> type;
};

template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  struct function_signature<Result(*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7> type;
};

template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  struct function_signature<Result(*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8> type;
};

template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
  struct function_signature<Result(*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9> type;
};


// specializations for ptr to member function
template<typename Result, typename Class>
  struct function_signature<Result(Class::*)()>
{
  typedef type_list<Result> type;
};

template<typename Result, typename Class, typename Arg1>
  struct function_signature<Result(Class::*)(Arg1)>
{
  typedef type_list<Result,Arg1> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2>
  struct function_signature<Result(Class::*)(Arg1,Arg2)>
{
  typedef type_list<Result,Arg1,Arg2> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2, typename Arg3>
  struct function_signature<Result(Class::*)(Arg1,Arg2,Arg3)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  struct function_signature<Result(Class::*)(Arg1,Arg2,Arg3,Arg4)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  struct function_signature<Result(Class::*)(Arg1,Arg2,Arg3,Arg4,Arg5)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  struct function_signature<Result(Class::*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  struct function_signature<Result(Class::*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
  struct function_signature<Result(Class::*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8> type;
};

template<typename Result, typename Class, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
  struct function_signature<Result(Class::*)(Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9)>
{
  typedef type_list<Result,Arg1,Arg2,Arg3,Arg4,Arg5,Arg6,Arg7,Arg8,Arg9> type;
};

