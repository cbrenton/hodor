#ifndef _HITDATA_H
#define _HITDATA_H


#include <Eigen/Dense>
using namespace Eigen;

class Geometry;

//A struct to hold various data resulting from an intersection.
struct HitData
{
    bool hit;

    Vector3f point;

    float t;

    Geometry *object;

};
#endif
