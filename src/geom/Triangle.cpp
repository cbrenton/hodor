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
   minX = min3(t_t.c1.x, t_t.c2.x, t_t.c3.x);
   maxX = max3(t_t.c1.x, t_t.c2.x, t_t.c3.x);
   minY = min3(t_t.c1.y, t_t.c2.y, t_t.c3.y);
   maxY = max3(t_t.c1.y, t_t.c2.y, t_t.c3.y);
   minZ = min3(t_t.c1.z, t_t.c2.z, t_t.c3.z);
   maxZ = max3(t_t.c1.z, t_t.c2.z, t_t.c3.z);
   vec3 c1 = vec3(minX, minY, minZ);
   vec3 c2 = vec3(maxX, maxY, maxZ);
   return Box(c1, c2);
}

int Triangle::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   float result = -1;
   float bBeta, bGamma, bT;

   mat3 A (
         t_t.c1.x-t_t.c2.x, t_t.c1.x-t_t.c3.x, ray.dir.x,
         t_t.c1.y-t_t.c2.y, t_t.c1.y-t_t.c3.y, ray.dir.y,
         t_t.c1.z-t_t.c2.z, t_t.c1.z-t_t.c3.z, ray.dir.z
         );
   float detA = determinant(A);

   mat3 baryT (
         t_t.c1.x-t_t.c2.x, t_t.c1.x-t_t.c3.x, t_t.c1.x-ray.point.x,
         t_t.c1.y-t_t.c2.y, t_t.c1.y-t_t.c3.y, t_t.c1.y-ray.point.y,
         t_t.c1.z-t_t.c2.z, t_t.c1.z-t_t.c3.z, t_t.c1.z-ray.point.z
         );

   bT = determinant(baryT) / detA;

   if (bT < 0)
   {
      result = 0;
   }
   else
   {
      mat3 baryGamma (
            t_t.c1.x-t_t.c2.x, t_t.c1.x-ray.point.x, ray.dir.x,
            t_t.c1.y-t_t.c2.y, t_t.c1.y-ray.point.y, ray.dir.y,
            t_t.c1.z-t_t.c2.z, t_t.c1.z-ray.point.z, ray.dir.z
            );

      bGamma = determinant(baryGamma) / detA;

      if (bGamma < 0 || bGamma > 1)
      {
         result = 0;
      }
      else
      {
         mat3 baryBeta (
               t_t.c1.x-ray.point.x, t_t.c1.x-t_t.c3.x, ray.dir.x,
               t_t.c1.y-ray.point.y, t_t.c1.y-t_t.c3.y, ray.dir.y,
               t_t.c1.z-ray.point.z, t_t.c1.z-t_t.c3.z, ray.dir.z
               );

         bBeta = determinant(baryBeta) / detA;

         if (bBeta < 0 || bBeta > 1 - bGamma)
         {
            result = 0;
         }
      }
   }

   if (result != 0)
   {
      result = bT;
   }
   *t = result;
   if (result > EPSILON)
   {
      data->hit = 1;
      data->point = ray.point + ray.dir * (*t);
      data->t = (*t);
      data->object = this;
      return 1;
   }
   return 0;
}

vec3 Triangle::getNormal(const vec3 & point)
{
   vec3 s1 = t_t.c2 - t_t.c1;
   vec3 s2 = t_t.c3 - t_t.c1;
   return cross(s1, s2);
}
