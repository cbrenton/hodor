/**
 * A geometry object representing a plane.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PLANE_H
#define _PLANE_H

#include "plane_t.h"
#include "Geometry.h"
#include "Box.h"

class Ray;
struct HitData;

class Plane : public Geometry
{
   public:
      Plane() {};

      Plane(Vector3f normal, float offset);

      // The plane_t struct representing the geometry object.
      plane_t p_t;

      // Gets the bounding box of the current geometry object.
      Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
      int hit(const Ray & ray, float *t, HitData *data = NULL, float minT = 0.0, float maxT = MAX_DIST);

      // Returns the normal of the current geometry object at the specified point.
      Vector3f getNormal(Vector3f point);

      inline void debug()
      {
         cout << "Plane: <" << p_t.normal.x() << ", " << p_t.normal.y() << ", " << p_t.normal.z() << ">, " << p_t.offset << endl;
      }

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};
#endif
