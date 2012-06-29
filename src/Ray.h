/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _RAY_H
#define _RAY_H

#include <iostream>
#include <glm/glm.hpp>

using namespace std;

class Ray
{
   public:
      Ray() {};

      Ray(glm::vec3 _point, glm::vec3 _dir);

      // The origin of the ray.
      glm::vec3 point;

      // The direction of the ray.
      glm::vec3 dir;

      inline void debug()
      {
         cout << "Ray: <" << point.x << ", " << point.y << ", " << point.z << ">" << endl;
         cout << "\t<" << dir.x << ", " << dir.y << ", " << dir.z << ">" << endl;
      }

};
#endif
