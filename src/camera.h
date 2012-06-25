/**
 * Sterling Hirsh
 * Camera Class
 * 4/3/11
 */

#ifndef _CAMERA_H
#define _CAMERA_H

#include <iostream>
#include "glm/glm.hpp"
#include "globals.h"

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
         std::cout << "Camera: ";
         mPRLN_VEC(location);
         mPRLN_VEC(up);
         mPRLN_VEC(right);
         mPRLN_VEC(look_at);
      }
};

#endif
