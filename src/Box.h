/**
 * A geometry object representing a box.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _BOX_H
#define _BOX_H

#include "box_t.h"
#include "Geometry.h"
#include "Plane.h"

class Ray;
struct HitData;

class Box : public Geometry
{
   public:
      Box() {};

      Box(Vector3f c1, Vector3f c2);

      // The box_t struct representing the geometry object.
      box_t b_t;

      // Gets the bounding box of the current geometry object.
      Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
      int hit(const Ray & ray, float *t, HitData *data = NULL, float minT = 0.0, float maxT = MAX_DIST);

      // Returns the normal of the current geometry object at the specified point.
      Vector3f getNormal(const Vector3f & point);

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};
#endif
