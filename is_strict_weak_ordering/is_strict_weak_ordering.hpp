#pragma once

// refer to http://en.wikipedia.org/wiki/Strict_weak_ordering#Properties

template<typename Iterator, typename Predicate>
bool has_irreflexivity(Iterator first, Iterator last, Predicate pred)
{
  for(; first != last; ++first)
  {
    if(pred(*first, *first)) return false;
  }

  return true;
}

template<typename Iterator, typename Predicate>
bool has_asymmetry(Iterator first, Iterator last, Predicate pred)
{
  for(Iterator x = first; x != last; ++x)
  {
    for(Iterator y = first; y != last; ++y)
    {
      if(x != y)
      {
        if(pred(*x,*y))
        {
          if(pred(*y,*x)) return false;
        }
      }
    }
  }

  return true;
}

template<typename Iterator, typename Predicate>
bool has_transitivity(Iterator first, Iterator last, Predicate pred)
{
  for(Iterator x = first; x != last; ++x)
  {
    for(Iterator y = first; y != last; ++y)
    {
      if(pred(*x,*y))
      {
        for(Iterator z = first; z != last; ++z)
        {
          if(pred(*y,*z))
          {
            if(!pred(*x,*z)) return false;
          }
        }
      }
    }
  }

  return true;
}

template<typename Iterator, typename Predicate>
bool has_transitivity_of_equivalence(Iterator first, Iterator last, Predicate pred)
{
  for(Iterator x = first; x != last; ++x)
  {
    for(Iterator y = first; y != last; ++y)
    {
      if(pred(*x,*y))
      {
        for(Iterator z = first; z != last; ++z)
        {
          if(!(pred(*x,*z) || pred(*z,*y))) return false;
        }
      }
    }
  }

  return true;
}

template<typename Iterator, typename Predicate>
bool is_strict_weak_ordering(Iterator first, Iterator last, Predicate pred)
{
  return has_irreflexivity(first,last,pred) && has_asymmetry(first,last,pred) && has_transitivity(first,last,pred) && has_transitivity_of_equivalence(first,last,pred);
}

