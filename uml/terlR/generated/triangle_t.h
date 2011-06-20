#ifndef _TRIANGLE_T_H
#define _TRIANGLE_T_H


#include <Eigen/Dense>
using namespace Eigen;

//A struct representing a triangle.
struct triangle_t
{
    //The first corner of the triangle.
    Vector3f location;

    //The second corner of the triangle.
    Vector3f c2;

    //The third corner of the triangle.
    Vector3f c3;

};
#endif
