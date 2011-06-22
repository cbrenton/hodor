/**
 * A geometry object representing a triangle.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include "triangle_t.h"
#include "Geometry.h"
#include "Box.h"

class Ray;
struct HitData;

class Triangle : public Geometry
{
   public:
      // The triangle_t struct representing the geometry object.
      triangle_t t_t;

      // Gets the bounding box of the current geometry object.
      Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
      int hit(const Ray & ray, float *t, HitData *data, float minT, float maxT);

      // Returns the normal of the current geometry object at the specified point.
      Vector3f getNormal(const Vector3f & point);

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};
#endif
