#ifndef _SPHERE_T_H
#define _SPHERE_T_H


#include <Eigen/Dense>
using namespace Eigen;

//A struct representing a sphere.
struct sphere_t
{
    //The radius of the sphere.
    float radius;


  private:
    //The location in world space of the sphere.
    Vector3f location;

};
#endif
