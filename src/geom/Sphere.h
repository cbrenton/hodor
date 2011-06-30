/**
 * A geometry object representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _SPHERE_H
#define _SPHERE_H

#include "structs/sphere_t.h"
#include "geom/Geometry.h"
#include "geom/Box.h"

class Ray;
struct HitData;

class Sphere : public Geometry
{
   public:
      // The sphere_t struct representing the geometry object.
      sphere_t s_t;

      // Gets the bounding box of the current geometry object.
      Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
      int hit(Ray & ray, float *t, HitData *data = NULL, float minT = 0.0, float maxT = MAX_DIST);

      // Returns the normal of the current geometry object at the specified point.
      vec3_t getNormal(vec3_t & point);

      inline void debug()
      {
         cout << "Sphere: <" << s_t.location.x() << ", " << s_t.location.y() << ", " << s_t.location.z() << ">, " << s_t.radius << endl;
      }

};
#endif
