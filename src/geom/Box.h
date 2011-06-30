/**
 * A geometry object representing a box.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _BOX_H
#define _BOX_H

#include "structs/box_t.h"
#include "geom/Geometry.h"
#include "Plane.h"

class ray_t;
struct hit_t;

class Box : public Geometry
{
   public:
      Box() {};

      Box(vec3_t c1, vec3_t c2);

      // The box_t struct representing the geometry object.
      box_t b_t;

      // Gets the bounding box of the current geometry object.
      Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input hit_t object.
      int hit(ray_t & ray, float *t, hit_t *data = NULL, float minT = 0.0, float maxT = MAX_DIST);

      // Returns the normal of the current geometry object at the specified point.
      vec3_t getNormal(vec3_t & point);

};
#endif
