/**
 * A geometry object representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Sphere.h"
#include "Ray.h"
#include "structs/HitData.h"

// Gets the bounding box of the current geometry object.
Box Sphere::bBox()
{
   return Box();
}

int Sphere::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   // Optimized algorithm courtesy of "Real-Time Rendering, Third Edition".
   vec3 l = s_t.location - ray.point;
   float s = dot(l, ray.dir);
   float l2 = dot(l, l);
   float r2 = s_t.radius * s_t.radius;
   if (s < 0 && l2 > r2)
   {
      return 0;
   }
   float m2 = l2 - (s*s);
   if (m2 > r2)
   {
      return 0;
   }
   float q = sqrt(r2 - m2);
   if (l2 > r2)
   {
      data->t = s - q;
      *t = s - q;
   }
   else
   {
      data->t = s + q;
      *t = s + q;
   }
   data->point = ray.point + ray.dir * (*t);
   data->object = this;
   if (l2 < r2)
   {
      data->hit = -1;
      return -1;
   }
   else
   {
      data->hit = 1;
      return 1;
   }
}

vec3 Sphere::getNormal(const vec3 & point)
{
   vec3 n = point - s_t.location;
   n = normalize(n);
   return n;
}
