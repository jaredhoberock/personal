#include "cta.hpp"
#include <ucontext.h>
#include <utility>

class ucontext_cta
  : public cta
{
  public:
    template<typename Function>
      ucontext_cta(int num_threads, Function f)
        : cta()
    {
      thread_state.clear();

      // arguments to makecontext
      // XXX each thread needs its own copy of f -- we should not refer to the same functor
      void (*exec)(std::pair<ucontext_cta*,Function> *) = exec_thread<Function>;
      std::pair<ucontext_cta*,Function> exec_parms = std::make_pair(this, f);

      // save the return state
      state join_state;
      getcontext(&join_state);
      if(thread_state.empty())
      {
        thread_state.clear();
        thread_state.resize(num_threads);

        for(int i = 0; i < thread_state.size(); ++i)
        {
          getcontext(&thread_state[i]);
          thread_state[i].uc_link = &join_state;
          thread_state[i].uc_stack.ss_sp = thread_state[i].stack;
          thread_state[i].uc_stack.ss_size = sizeof(thread_state[i].stack);
          makecontext(&thread_state[i], (void(*)())exec, 1, &exec_parms);
        }

        // start thread 0
        set_current_thread_id(0);
        setcontext(&thread_state[0]);
      }

      // null the current cta
      this_thread_group::__singleton = 0;
    }

    virtual int num_threads()
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

    template<typename Function>
      static void exec_thread(std::pair<ucontext_cta*,Function> *parms)
    {
      parms->second();

      parms->first->at_exit();
    }

    struct state
      : ucontext_t
    {
      char stack[1<<16];
    };

    std::vector<state> thread_state;
};

