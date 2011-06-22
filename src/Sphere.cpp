/**
 * A geometry object representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Sphere.h"
#include "Ray.h"
#include "HitData.h"

// Gets the bounding box of the current geometry object.
Box Sphere::bBox()
{
  return Box();
}

int Sphere::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
  return 0;
}

Vector3f Sphere::getNormal(const Vector3f & point)
{
  return Vector3f();
}
