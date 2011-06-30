/**
 * The abstract class representing a geometry object in a scene. Hit detection per object is done here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _GEOMETRY_H
#define _GEOMETRY_H

#include "structs/pigment_t.h"
#include "structs/finish_t.h"
#include "geom/Transformation.h"
#include <iostream>
#include "structs/vector.h"

#define MAX_DIST 10000.0f

using namespace std;

class ray_t;
class Box;
struct hit_t;

class Geometry
{
   public:
      // The pigment of the geometry object.
      pigment_t p;

      // The finish of the geometry object.
      finish_t f;

      // Gets the bounding box of the current geometry object.
      virtual Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input hit_t object.
      virtual int hit(ray_t & ray, float *t, hit_t *data = NULL, float minT = 0.0, float maxT = MAX_DIST); 

      // Returns the normal of the current geometry object at the specified point.
      virtual vec3_t getNormal(vec3_t & point);

      virtual void debug() {};

      //void addTransformation(Transform<float, 3, Affine> t);

      //Transform<float, 3, Affine> transform;

};
#endif
