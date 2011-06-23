/**
 * A geometry object representing a triangle.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Triangle.h"
#include "Ray.h"
#include "HitData.h"

// Gets the bounding box of the current geometry object.
Box Triangle::bBox()
{
   return Box();
}

int Triangle::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   return 0;
}

Vector3f Triangle::getNormal(const Vector3f & point)
{
   return Vector3f();
}
