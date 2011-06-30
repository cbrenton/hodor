
#include "geom/Mesh2.h"
#include "structs/ray_t.h"
#include "structs/hit_t.h"

// Gets the bounding box of the current geometry object.
Box Mesh2::bBox()
{
   Box result;
   return result;
}

int Mesh2::hit(ray_t & ray, float *t, hit_t *data, float minT, float maxT)
{
   return 0;
}

vec3_t Mesh2::getNormal(vec3_t & point)
{
   return vec3_t();
}
