
#include "geom/Mesh2.h"
#include "Ray.h"
#include "structs/HitData.h"

// Gets the bounding box of the current geometry object.
Box Mesh2::bBox()
{
   return Box();
}

int Mesh2::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   return 0;
}

Vector3f Mesh2::getNormal(const Vector3f & point)
{
   return Vector3f();
}
