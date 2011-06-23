/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _RAY_H
#define _RAY_H

#include <iostream>
#include <Eigen/Dense>

using namespace std;

class Ray
{
   public:
      Ray() {};

      Ray(Eigen::Vector3f _point, Eigen::Vector3f _dir);

      // The origin of the ray.
      Eigen::Vector3f point;

      // The direction of the ray.
      Eigen::Vector3f dir;

      inline void debug()
      {
         cout << "Ray: <" << point.x() << ", " << point.y() << ", " << point.z() << ">" << endl;
         cout << "\t<" << dir.x() << ", " << dir.y() << ", " << dir.z() << ">" << endl;
      }

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};
#endif
