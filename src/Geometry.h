/**
 * The abstract class representing a geometry object in a scene. Hit detection per object is done here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _GEOMETRY_H
#define _GEOMETRY_H

#include "Pigment.h"
#include "Finish.h"
#include <Eigen/Dense>

using namespace Eigen;

class Ray;
class Box;
struct HitData;

class Geometry
{
  public:
    //The pigment of the geometry object.
    Pigment pigment;

    //The finish of the geometry object.
    Finish finish;

    //Gets the bounding box of the current geometry object.
    virtual Box bBox();

    //Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
    virtual int hit(const Ray & ray, float *t, HitData *data, float minT, float maxT);

    //Returns the normal of the current geometry object at the specified point.
    virtual Vector3f getNormal(const Vector3f & point);

    void addTransformation(Transform<float, 3, Affine> t);

    Transform<float, 3, Affine> transform;

};
#endif
