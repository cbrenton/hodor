/**
 * A geometry object representing a plane.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Plane.h"
#include "Ray.h"
#include "structs/HitData.h"

Plane::Plane(vec3 normal, float offset)
{
   p_t.normal = normal;
   p_t.offset = offset;
}

// Gets the bounding box of the current geometry object.
Box Plane::bBox()
{
   return Box();
}

int Plane::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   float denominator = dot(ray.dir, p_t.normal);
   if (denominator == 0.0)
   {
      return 0;
   }
   vec3 p = p_t.normal * p_t.offset;
   vec3 pMinusL = p - ray.point;
   float numerator = dot(pMinusL, p_t.normal);
   *t = numerator / denominator;
   if (*t >= minT && *t <= maxT)
   {
      data->hit = 1;
      data->point = ray.point + ray.dir * (*t);
      data->t = (*t);
      data->object = this;
      return 1;
   }
   return 0;
}

vec3 Plane::getNormal(const vec3 & point)
{
   return p_t.normal;
}
