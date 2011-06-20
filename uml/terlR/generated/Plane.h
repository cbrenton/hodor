#ifndef _PLANE_H
#define _PLANE_H


#include "plane_t.h"
#include "Geometry.h"
#include "Box.h"
#include <Eigen/Dense>
using namespace Eigen;

class Ray;
struct HitData;

//A geometry object representing a plane.
class Plane : public Geometry
{
  protected:
    //The plane_t struct representing the geometry object.
    plane_t p;


  public:
    //Gets the bounding box of the current geometry object.
    virtual Box bBox();

    //Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
    virtual int hit(const Ray & ray, float *t, const HitData & *data, float minT, float maxT);

    //Returns the normal of the current geometry object at the specified point.
    virtual Vector3f getNormal(const Vector3f & point);

};
#endif
