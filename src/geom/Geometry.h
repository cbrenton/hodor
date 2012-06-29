/**
 * The abstract class representing a geometry object in a scene. Hit detection per object is done here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _GEOMETRY_H
#define _GEOMETRY_H

#include "structs/Pigment.h"
#include "structs/Finish.h"
#include <iostream>
#include <glm/glm.hpp>

#define MAX_DIST 10000.0f

using namespace glm;
using namespace std;

class Ray;
class Box;
struct HitData;

class Geometry
{
   public:
      // The pigment of the geometry object.
      Pigment p;

      // The finish of the geometry object.
      Finish f;

      // Gets the bounding box of the current geometry object.
      virtual Box bBox();

      // Determines whether the input ray intersects the current geometry object. If it does not, returns 0. If it does, returns -1 if hit from within the object, or 1 if hit from outside the object, and correctly populates the fields of the input HitData object.
      virtual int hit(const Ray & ray, float *t, HitData *data = NULL, float minT = 0.0, float maxT = MAX_DIST); 

      // Returns the normal of the current geometry object at the specified point.
      virtual glm::vec3 getNormal(const glm::vec3 & point);

      virtual void debug() {};

      //void addTransformation(Transform<float, 3, Affine> t);

      //Transform<float, 3, Affine> transform;

};
#endif
