/**
 * The abstract class representing a geometry object in a scene. Hit detection per object is done here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Geometry.h"
#include "Ray.h"
#include "Box.h"
#include "HitData.h"

Box Geometry::bBox()
{
  return Box();
}

int Geometry::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
  return 0;
}

Vector3f Geometry::getNormal(const Vector3f & point)
{
  return Vector3f();
}

void Geometry::addTransformation(Transform<float, 3, Affine> t)
{
}
