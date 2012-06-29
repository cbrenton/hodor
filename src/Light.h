/**
 * Light - not very heavy.
 * Sterling Hirsh
 * 4/5/11
 */

#ifndef _LIGHT_H
#define _LIGHT_H

#include <iostream>
#include <stdio.h>
#include <glm/glm.hpp>
using glm::vec3;

class Light {
   public:
      Light(glm::vec3 loc, double red, double green, double blue);
      glm::vec3 location;
      double r;
      double g;
      double b;
      inline void debug()
      {
         printf("Light: (r,g,b) (%f, %f, %f)\n", r, g, b);
      }
};

#endif
