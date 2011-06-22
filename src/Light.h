/**
 * Light - not very heavy.
 * Sterling Hirsh
 * 4/5/11
 */

#ifndef _LIGHT_H
#define _LIGHT_H

#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
using Eigen::Vector3f;

class Light {
   public:
      Light(Vector3f loc, double red, double green, double blue);
      Vector3f location;
      double r;
      double g;
      double b;
      inline void debug()
      {
         printf("Light: (r,g,b) (%f, %f, %f)\n", r, g, b);
      }
};

#endif
