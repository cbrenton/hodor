/**
 * A geometry object representing a plane.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Plane.h"
#include "Ray.h"
#include "structs/HitData.h"

Plane::Plane(vec3_t normal, float offset)
{
   p_t.normal = normal;
   p_t.offset = offset;
}

// Gets the bounding box of the current geometry object.
Box Plane::bBox()
{
   Box result;
   return result;
}

int Plane::hit(Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   float denominator = ray.dir.dot(p_t.normal);
   if (denominator == 0.0)
   {
      return 0;
   }
   vec3_t p = p_t.normal * p_t.offset;
   vec3_t pMinusL = p - ray.point;
   float numerator = pMinusL.dot(p_t.normal);
   *t = numerator / denominator;
   if (*t >= minT && *t <= maxT)
   {
      data->hit = 1;
      data->point = ray.dir * (*t);
      data->point += ray.point;
      data->t = (*t);
      data->object = this;
      return 1;
   }
   return 0;
}

vec3_t Plane::getNormal(vec3_t & point)
{
   return p_t.normal;
}
