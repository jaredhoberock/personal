#pragma once

#include "../thread_group.hpp"
#include "closure_cpp11.hpp"
#include <utility>
#include <vector>

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE_NEEDS_UNDEF
#define _XOPEN_SOURCE
#endif // _XOPEN_SOURCE

#include <ucontext.h>

#ifdef _XOPEN_SOURCE_NEEDS_UNDEF
#undef _XOPEN_SOURCE
#undef _XOPEN_SOURCE_NEEDS_UNDEF
#endif // _XOPEN_SOURCE_NEEDS_UNDEF

namespace test
{
namespace detail
{


class ucontext_thread_group
  : public thread_group
{
  private:
    static const std::size_t stack_size = 1<<16;

  public:
    template<typename Function, typename... Args>
      ucontext_thread_group(int id, int num_threads, Function &&f, Args&&... args)
        : thread_group(id)
    {
      exec(num_threads, std::forward<Function>(f), std::forward<Args>(args)...);
    }

    virtual int size()
    {
      return thread_state.size();
    }

    void barrier()
    {
      // switch to next thread
      int old_thread_id = set_next_current_thread_id();
      swapcontext(&thread_state[old_thread_id], &thread_state[current_thread_id()]);
    }

  private:
    void at_exit()
    {
      barrier();
    }

    template<typename Function, typename... Args>
      void exec(std::size_t num_threads, Function &&f, Args&&... args)
    {
      if(num_threads)
      {
        // begin by making a closure
        auto closure = detail::make_closure(std::forward<Function>(f),std::forward<Args>(args)...);
        typedef decltype(closure) closure_type;

        // make a copy of the parameters for each thread
        // for arguments to makecontext
        typedef std::pair<ucontext_thread_group*,closure_type> exec_parms_t;
        void (*exec)(exec_parms_t *) = exec_thread<closure_type>;
        std::vector<exec_parms_t> exec_parms(num_threads, std::make_pair(this,closure));

        // save the return state
        state join_state;
        getcontext(&join_state);
        if(thread_state.empty())
        {
          thread_state.resize(num_threads);

          for(int i = 0; i < thread_state.size(); ++i)
          {
            getcontext(&thread_state[i]);
            thread_state[i].uc_link = &join_state;
            thread_state[i].uc_stack.ss_sp = thread_state[i].stack;
            thread_state[i].uc_stack.ss_size = sizeof(thread_state[i].stack);
            makecontext(&thread_state[i], (void(*)())exec, 1, &exec_parms[i]);
          }

          // start thread 0
          set_current_thread_id(0);
          setcontext(&thread_state[0]);
        }

        // when we've reached this point, all the threads in the group have terminated
      } // end if

      // null the current thread_group
      this_thread_group::__singleton = 0;
    }

    template<typename Function>
      static void exec_thread(std::pair<ucontext_thread_group*,Function> *parms)
    {
      try
      {
        parms->second();
      }
      catch(...)
      {
        // XXX ignore any exception
      }

      parms->first->at_exit();
    }

    struct state
      : ucontext_t
    {
      char stack[stack_size];
    };

    std::vector<state> thread_state;
};


} // end detail
} // end thread_group

