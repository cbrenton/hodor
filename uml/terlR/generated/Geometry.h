#ifndef _GEOMETRY_H
#define _GEOMETRY_H


#include "Pigment.h"
#include "Finish.h"
#include <Eigen/Dense>
using namespace Eigen;
#include "Box.h"

class Ray;
struct HitData;

//The abstract class representing a geometry object in a scene. Hit detection per object is done here.
class Geometry
{
  public:
    //The pigment of the geometry object.
    Pigment pigment;

    //The finish of the geometry object.
    Finish finish;

    //The location of the geometry object.
    Vector3f location;

    //Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
    virtual int hit(const Ray & ray, float *t, const HitData & *data, float minT, float maxT) = 0;

    //Returns the normal of the current geometry object at the specified point.
    virtual Vector3f getNormal(const Vector3f & point) = 0;

    //Gets the bounding box of the current geometry object.
    virtual Box bBox() = 0;

    Eigen::Matrix3f transform;

};
#endif
