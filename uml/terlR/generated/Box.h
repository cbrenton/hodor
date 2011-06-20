#ifndef _BOX_H
#define _BOX_H


#include "box_t.h"
#include "Geometry.h"
#include <Eigen/Dense>
using namespace Eigen;

class Ray;
struct HitData;

//A geometry object representing a box.
class Box : public Geometry
{
  protected:
    //The box_t struct representing the geometry object.
    box_t b;


  public:
    //Gets the bounding box of the current geometry object.
    virtual Box bBox();

    //Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
    virtual int hit(const Ray & ray, float *t, const HitData & *data, float minT, float maxT);

    //Returns the normal of the current geometry object at the specified point.
    virtual Vector3f getNormal(const Vector3f & point);

};
#endif
