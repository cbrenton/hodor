/**
 * Sterling Hirsh
 * Camera Class
 * 4/3/11
 */

#ifndef _CAMERA_H
#define _CAMERA_H

#include <iostream>
#include "glm/glm.hpp"

class Camera {
   public:
      Camera() {};
      Camera(glm::vec3 _loc, glm::vec3 _up, glm::vec3 _right, glm::vec3 _look_at);
      glm::vec3 location;
      glm::vec3 up;
      glm::vec3 right;
      glm::vec3 look_at;
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
