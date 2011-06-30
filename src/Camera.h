/**
 * Sterling Hirsh
 * Camera Class
 * 4/3/11
 */

#ifndef _CAMERA_H
#define _CAMERA_H

#include <iostream>
#include <structs/vector.h>

class Camera {
   public:
      Camera() {};
      Camera(vec3_t _loc, vec3_t _up, vec3_t _right, vec3_t _look_at);
      vec3_t location;
      vec3_t up;
      vec3_t right;
      vec3_t look_at;
      inline void debug()
      {
         std::cout << "Camera: <" << location.x() << ", " << location.y() <<
            ", " << location.z() << ">" << std::endl;
         std::cout << "\t<" << up.x() << ", " << up.y() <<
            ", " << up.z() << ">" << std::endl;
         std::cout << "\t<" << right.x() << ", " << right.y() <<
            ", " << right.z() << ">" << std::endl;
         std::cout << "\t<" << look_at.x() << ", " << look_at.y() <<
            ", " << look_at.z() << ">" << std::endl;
      }
};

#endif
