/**
 * A geometry object representing a triangle mesh (mesh2 in the povray format).
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _MESH2_H
#define _MESH2_H

#include "Triangle.h"
#include "Box.h"
#include "Geometry.h"

class Ray;
struct HitData;

class Mesh2 : public Geometry
{
  protected:
    //A vector containing the triangles represented by the mesh.
    Triangle faces;

  public:
    //Gets the bounding box of the current geometry object.
    Box bBox();

    //Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
    int hit(const Ray & ray, float *t, const HitData & *data, float minT, float maxT);

    //Returns the normal of the current geometry object at the specified point.
    Vector3f getNormal(const Vector3f & point);

};
#endif
