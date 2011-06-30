/**
 * A geometry object representing a triangle.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Triangle.h"
#include "Ray.h"
#include "structs/HitData.h"
#include "Globals.h"

// Gets the bounding box of the current geometry object.
Box Triangle::bBox()
{
   float minX, minY, minZ, maxX, maxY, maxZ;
   minX = min3(t_t.c1.x(), t_t.c2.x(), t_t.c3.x());
   maxX = max3(t_t.c1.x(), t_t.c2.x(), t_t.c3.x());
   minY = min3(t_t.c1.y(), t_t.c2.y(), t_t.c3.y());
   maxY = max3(t_t.c1.y(), t_t.c2.y(), t_t.c3.y());
   minZ = min3(t_t.c1.z(), t_t.c2.z(), t_t.c3.z());
   maxZ = max3(t_t.c1.z(), t_t.c2.z(), t_t.c3.z());
   vec3_t c1 = vec3_t(minX, minY, minZ);
   vec3_t c2 = vec3_t(maxX, maxY, maxZ);
   Box result(c1, c2);
   return result;
}

int Triangle::hit(Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   return 0;
}

vec3_t Triangle::getNormal(vec3_t & point)
{
   vec3_t s1 = t_t.c2 - t_t.c1;
   vec3_t s2 = t_t.c3 - t_t.c1;
   s1.cross(s1, s2);
   return s1;
}
