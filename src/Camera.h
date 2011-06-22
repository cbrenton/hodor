/**
 * Sterling Hirsh
 * Camera Class
 * 4/3/11
 */

#ifndef _CAMERA_H
#define _CAMERA_H

#include <iostream>
#include <Eigen/Dense>
using Eigen::Vector3f;

class Camera {
   public:
      Camera(Vector3f _loc, Vector3f _up, Vector3f _right, Vector3f _look_at);
      Vector3f location;
      Vector3f up;
      Vector3f right;
      Vector3f look_at;
      inline void debug()
      {
        std::cout << "Camera: <" << location(0) << ", " << location(1) <<
          ", " << location(2) << ">" << std::endl;
      }
};

#endif
