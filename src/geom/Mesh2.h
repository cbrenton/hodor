/**
 * A geometry object representing a triangle mesh (mesh2 in the povray format).
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _MESH2_H
#define _MESH2_H

#include "geom/Triangle.h"
#include "geom/Box.h"
#include "geom/Geometry.h"
#include <vector>

class ray_t;
struct hit_t;

class Mesh2 : public Geometry
{
   protected:
      // A vector containing the triangles represented by the mesh.
      std::vector<Triangle> faces;

   public:
      // Gets the bounding box of the current geometry object.
      Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input hit_t object.
      int hit(ray_t & ray, float *t, hit_t *data = NULL, float minT = 0.0, float maxT = MAX_DIST);

      // Returns the normal of the current geometry object at the specified point.
      vec3_t getNormal(vec3_t & point);

};
#endif
