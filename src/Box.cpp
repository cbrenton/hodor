/**
 * A geometry object representing a box.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Box.h"
#include "Ray.h"
#include "HitData.h"

Box::Box(Vector3f c1, Vector3f c2)
{
   b_t.c1 = c1;
   b_t.c2 = c2;
}

// Gets the bounding box of the current geometry object.
Box Box::bBox()
{
   return Box();
}

int Box::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   return 0;
}

Vector3f Box::getNormal(const Vector3f & point)
{
   return Vector3f();
}
