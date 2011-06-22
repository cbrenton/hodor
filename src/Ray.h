/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _RAY_H
#define _RAY_H

#include <Eigen/Dense>

class Ray
{
   public:
      // The origin of the ray.
      Eigen::Vector3f point;

      // The direction of the ray.
      Eigen::Vector3f dir;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};
#endif
