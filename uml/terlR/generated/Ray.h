#ifndef _RAY_H
#define _RAY_H


#include <Eigen/Dense>
using namespace Eigen;

class Ray
{
  public:
    //The origin of the ray.
    Vector3f point;

    //The direction of the ray.
    Vector3f dir;

};
#endif
