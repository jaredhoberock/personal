#pragma once

#include "../thread_group.hpp"

namespace test
{
namespace detail
{

class serial_thread_group
  : public thread_group
{
  public:
    template<typename Function, typename Tuple>
      inline serial_thread_group(int id, int num_threads, Function f, Tuple args)
        : thread_group(id),
          m_size(num_threads)
    {
      exec(f, args);
    }

    inline int size()
    {
      return m_size;
    }

    inline void barrier() {}

  private:

    template<typename Function, typename Tuple>
      inline void exec(Function f, Tuple args)
    {
      // ensure sure that no virtual function dispatch occurs for size()
      const size_t sz = serial_thread_group::size();

      for(std::size_t thread_id = 0;
          thread_id != sz;
          ++thread_id)
      {
        // XXX this is pretty expensive
        set_current_thread_id(thread_id);

        // make & call closure
        detail::make_closure(f,args)();
      }

      // null the current thread_group
      this_thread_group::__singleton = 0;
    }

    std::size_t m_size;
};

} // end detail
} // end test

