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
      Camera() {};
      Camera(Vector3f _loc, Vector3f _up, Vector3f _right, Vector3f _look_at);
      Vector3f location;
      Vector3f up;
      Vector3f right;
      Vector3f look_at;
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
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
