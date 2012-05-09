/**
 * Sterling Hirsh
 * Camera Class
 * 4/3/11
 */

#ifndef _CAMERA_H
#define _CAMERA_H

#include <iostream>
#include "vector.h"

class Camera {
   public:
      Camera() {};
      Camera(vec3 _loc, vec3 _up, vec3 _right, vec3 _look_at);
      vec3 location;
      vec3 up;
      vec3 right;
      vec3 look_at;
      inline void debug()
      {
         /*
         std::cout << "Camera: <" << location.x() << ", " << location.y() <<
            ", " << location.z() << ">" << std::endl;
         std::cout << "\t<" << up.x() << ", " << up.y() <<
            ", " << up.z() << ">" << std::endl;
         std::cout << "\t<" << right.x() << ", " << right.y() <<
            ", " << right.z() << ">" << std::endl;
         std::cout << "\t<" << look_at.x() << ", " << look_at.y() <<
            ", " << look_at.z() << ">" << std::endl;
            */
      }
};

#endif
