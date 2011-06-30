/**
 * A geometry object representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Sphere.h"
#include "structs/ray_t.h"
#include "structs/hit_t.h"

// Gets the bounding box of the current geometry object.
Box Sphere::bBox()
{
   Box result;
   return result;
}

int Sphere::hit(ray_t & ray, float *t, hit_t *data, float minT, float maxT)
{
   return 0;
}

vec3_t Sphere::getNormal(vec3_t & point)
{
   vec3_t n = point - s_t.location;
   n.normalize();
   return n;
}
