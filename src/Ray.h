/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _RAY_H
#define _RAY_H

#include <iostream>
#include "structs/vector.h"

using namespace std;

class Ray
{
   public:
      Ray() {};

      Ray(vec3_t _point, vec3_t _dir);

      // The origin of the ray.
      vec3_t point;

      // The direction of the ray.
      vec3_t dir;

      inline void debug()
      {
         cout << "Ray: <" << point.x() << ", " << point.y() << ", " << point.z() << ">" << endl;
         cout << "\t<" << dir.x() << ", " << dir.y() << ", " << dir.z() << ">" << endl;
      }

};
#endif
