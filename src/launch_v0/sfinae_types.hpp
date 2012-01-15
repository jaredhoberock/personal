#pragma once

struct sfinae_types
{
  typedef char                      yes_type;
  typedef struct {char array[2]; }  no_type;
};

