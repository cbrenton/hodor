/**
 * Sterling Hirsh
 * Camera Class
 * 4/3/11
 */

#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <iostream>
#include <Eigen/Dense>
using Eigen::Vector3f;

class Camera {
   public:
      Camera(std::istream& input);
      Vector3f location;
      Vector3f up;
      Vector3f right;
      Vector3f look_at;
      inline void debug()
      {
         printf("Camera: <%f, %f, %f>\n", location.x(), location.y(), location.z());
      }
};

#endif
