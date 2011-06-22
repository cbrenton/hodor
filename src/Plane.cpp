/**
 * A geometry object representing a plane.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Plane.h"
#include "Ray.h"
#include "HitData.h"

//Gets the bounding box of the current geometry object.
Box Plane::bBox() {
  return Box();
}

int Plane::hit(Ray ray, float *t, HitData *data, float minT, float maxT)
{
  return 0;
}

Vector3f Plane::getNormal(Vector3f point)
{
  return Vector3f();
}
