
#include "Mesh2.h"
#include "Ray.h"
#include "HitData.h"

//Gets the bounding box of the current geometry object.
Box Mesh2::bBox() {
  return Box();
}

int Mesh2::hit(Ray ray, float *t, HitData *data, float minT, float maxT)
{
  return 0;
}

Vector3f Mesh2::getNormal(Vector3f point)
{
  return Vector3f();
}
