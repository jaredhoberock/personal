#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <limits>
#include <utility>

typedef std::pair<double,double> point;

// a node in a 2d kd tree
struct node
{
  int id;
  point position;
  bool split_axis;

  friend std::istream &operator>>(std::istream &is, topic &t)
  {
    return is >> t.id >> t.position.first >> t.position.second;
  }

  friend std::ostream &operator<<(std::ostream &os, const topic &t)
  {
    return os << t.id << " " << t.position.first << " " << t.position.second;
  }
};

struct question
{
  int id;
  std::vector<int> associated_topics;

  friend std::istream &operator>>(std::istream &is, question &q)
  {
    is >> q.id;

    int num_topics = 0;
    is >> num_topics;

    q.associated_topics.resize(num_topics);
    std::copy_n(std::istream_iterator<int>(is), q.associated_topics.size(), q.associated_topics.begin());

    return is;
  }
};

struct box
{
  // constructs an empty box
  box()
    : ll(std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
      ur(std::numeric_limits<double>::min(), std::numeric_limits<double>::min())
  {}

  box(point ll, point ur)
    : ll(ll), ur(ur)
  {}

  // lower left & upper right corners of the box
  point ll, ur;

  bool larger_axis() const
  {
    return (ur.second - ll.first) > (ur.first - ll.first);
  }

  // returns a new box containing this box and x
  box add_point(point x) const
  {
    box result = *this;

    if(x.first < result.ll.first)
    {
      result.ll.first = x.first;
    }
    else if(x.first > result.ur.first)
    {
      result.ur.first = x.first;
    }

    if(x.second < result.ll.second)
    {
      result.ll.second = x.second;
    }
    else if(x.second > result.ur.second)
    {
      result.ur.second = x.second;
    }

    return result;
  }
};

// returns a box containing b and n's position
box add_point(const box &b, const kdnode &n)
{
  return b.add_point(n.position);
}

template<unsigned int axis>
bool sort_nodes(const node &lhs, const node &rhs)
{
  return std::get<axis>(lhs.position) < std::get<axis>(rhs.position);
}

void spatial_sort(node *first, node *last)
{
  std::ptrdiff_t n = last - first;

  if(n > 1)
  {
    // choose a pivot
    node *mid = first + (n / 2);

    // find the bounds of the range
    box bounds = std::accumulate(first, last, box(), add_point);

    // choose the larger axis to split
    mid->split_axis = bounds.larger_axis();
    
    // sort on the larger axis
    if(split_axis)
    {
      std::nth_element(first, mid, last, sort_topics<1>);
    }
    else
    {
      std::nth_element(first, mid, last, sort_topics<0>);
    }

    // recurse
    spatial_sort(first, mid);
    spatial_sort(mid, last);
  }
}

// outputs the indices of the n nearest nodes to needle to result
// [haystack_first, haystack_last) is assumed to be organized as a kd tree
// n is assumed to be larger than haystack_last - haystack_first
void nearest_n(const node *haystack_first, const node *haystack_last,
               point needle, std::size_t n, int *result)
{
}

int main()
{
  int T = 0, Q = 0, N = 0;

  std::cin >> T >> Q >> N;

  // get topics
  std::vector<topic> topics(T);
  std::copy_n(std::istream_iterator<topic>(std::cin), T, topics.begin());

  std::copy(topics.begin(), topics.end(), std::ostream_iterator<topic>(std::cout, "\n"));
  std::cout << std::endl;

  // get questions
  std::vector<question> questions(Q);
  std::copy_n(std::istream_iterator<question>(std::cin), Q, questions.begin());

  // XXX eliminate irrelevant questions

  // get queries
  std::vector<point> topic_queries, question_queries;

  for(int i = 0; i < N; ++i)
  {
    char query_type = 0;
    std::cin >> query_type;

    point x;
    std::cin >> x.first >> x.second;

    if(query_type == 't')
    {
      topic_queries.push_back(x);
    }
    else if(query_type == 'q')
    {
      question_queries.push_back(x);
    }
  }

  // build a kdtree
  spatial_sort(topics.data(), topics.data() + topics.size());

  return 0;
}

