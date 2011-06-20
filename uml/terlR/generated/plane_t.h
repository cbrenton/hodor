#ifndef _PLANE_T_H
#define _PLANE_T_H


#include <Eigen/Dense>
using namespace Eigen;

//A struct representing a plane.
struct plane_t
{
    //The normal of the plane.
    Vector3f normal;

    //The plane offset ('d' in the plane's implicit equation).
    float offset;

};
#endif
