#pragma once

#include <cstddef>

template<typename Function, typename... Args>
void launch(std::size_t num_blocks, std::size_t num_threads_per_block, Function &&f, Args&&... args);

#include "launch_cpp11.inl"

