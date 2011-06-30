/**
 * Light - not very heavy.
 * Sterling Hirsh
 * 4/5/11
 */

#ifndef _LIGHT_H
#define _LIGHT_H

#include <iostream>
#include <stdio.h>
#include "structs/vector.h"

class Light {
   public:
      Light(vec3_t loc, double red, double green, double blue);
      vec3_t location;
      double r;
      double g;
      double b;
      inline void debug()
      {
         printf("Light: (r,g,b) (%f, %f, %f)\n", r, g, b);
      }
};

#endif
