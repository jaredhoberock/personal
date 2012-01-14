#pragma once

template<typename Function>
void launch(Function f);

template<typename Function, typename Arg1>
void launch(Function f, Arg1 arg1);

template<typename Function, typename Arg1, typename Arg2>
void launch(Function f, Arg1 arg1, Arg2 arg2);

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3);

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4);

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5);

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6);

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7);

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8);

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9);

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
void launch(Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9, Arg10 arg10);

#include "launch_cpp03.inl"

