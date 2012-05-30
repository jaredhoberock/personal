#pragma once

#include "../thread_group.hpp"
#include <ucontext.h>
#include <utility>
#include <vector>

namespace test
{
namespace detail
{

class serial_thread_group
  : public thread_group
{
  public:
    template<typename Function, typename... Args>
      inline serial_thread_group(int id, int num_threads, Function &&f, Args&&... args)
        : thread_group(id,num_threads)
    {
      exec(std::forward<Function>(f), std::forward<Args>(args)...);
    }

    inline void barrier() {}

  private:

    template<typename Function, typename... Args>
      inline void exec(Function &&f, Args&&... args)
    {
      const size_t sz = size();

      for(std::size_t thread_id = 0;
          thread_id != sz;
          ++thread_id)
      {
        // XXX this is pretty expensive
        set_current_thread_id(thread_id);

        // make & call closure
        detail::make_closure(std::forward<Function>(f),std::forward<Args>(args)...)();
      }

      // null the current thread_group
      this_thread_group::__singleton = 0;
    }
};

} // end detail
} // end test

