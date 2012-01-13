#pragma once

template<typename Function, typename... Args>
void launch(Function &&f, Args&&... args);

#include "launch.inl"

