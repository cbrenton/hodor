/**
 * A geometry object representing a plane.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PLANE_H
#define _PLANE_H

#include "structs/plane_t.h"
#include "geom/Geometry.h"
#include "geom/Box.h"

class Ray;
struct HitData;

class Plane : public Geometry
{
   public:
      Plane() {};

      Plane(vec3_t normal, float offset);

      // The plane_t struct representing the geometry object.
      plane_t p_t;

      // Gets the bounding box of the current geometry object.
      Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
      int hit(Ray & ray, float *t, HitData *data = NULL, float minT = 0.0, float maxT = MAX_DIST);

      // Returns the normal of the current geometry object at the specified point.
      vec3_t getNormal(vec3_t & point);

      inline void debug()
      {
         cout << "Plane: <" << p_t.normal.x() << ", " << p_t.normal.y() << ", " << p_t.normal.z() << ">, " << p_t.offset << endl;
      }

};
#endif
