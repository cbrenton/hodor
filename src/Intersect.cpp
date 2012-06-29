/**
 * Contains intersection data for all geometry structs.
 * @author Chris Brenton
 * @date 6/24/2011
 */

#include "Intersect.h"
#include "Globals.h"
#include <glm/glm.hpp>

using namespace glm;

int box_hit(const box_t & b_t, const Ray & ray, float *t, HitData *data)
{
   float tNear = MIN_T - 1;
   float tFar = MAX_DIST + 1;

   // For the pair of planes in each dimension.
   for (int i = 0; i < 3; i++)
   {
      // Component of the ray's direction in the current dimension.
      float xD = ray.dir[i];
      // Component of the ray's origin in the current dimension.
      float xO = ray.point[i];
      // Component of the mininum plane location in the current dimension.
      float xL = std::min(b_t.c1[i], b_t.c2[i]);
      // Component of the maxinum plane location in the current dimension.
      float xH = std::max(b_t.c1[i], b_t.c2[i]);
      // If direction in current dimension is 0, ray is parallel to planes.
      if (xD == 0)
      {
         // If ray origin is not between the planes.
         if (xO < xL || xO > xH)
         {
            return 0;
         }
      }
      // Else the ray is not parallel to the planes.
      else
      {
         // Calculate tMin and tMax.
         float t1 = (xL - xO) / xD;
         float t2 = (xH - xO) / xD;
         if (t1 > t2)
         {
            swap(t1, t2);
         }
         if (t1 > tNear)
         {
            tNear = t1;
         }
         if (t2 < tFar)
         {
            tFar = t2;
         }
         if (tNear > tFar)
         {
            return 0;
         }
         if (tFar < 0)
         {
            return 0;
         }
      }
   }
   *t = tNear;

   data->hit = 1;
   data->point = ray.point + ray.dir * (*t);
   data->t = (*t);
   data->hitType = BOX_HIT;
   if (b_t.f.reflection > 0.0)
   {
      data->reflect = new vec3();
      *data->reflect = mReflect(ray.dir, box_normal(b_t, data));
   }
   else
   {
      data->reflect = NULL;
   }
   return 1;
}

vec3 box_normal(const box_t & b_t, HitData *data)
{
   if (closeEnough(data->point.x, b_t.left.offset))
   {
      return b_t.left.normal;
   }
   if (closeEnough(data->point.x, b_t.right.offset))
   {
      return b_t.right.normal;
   }
   if (closeEnough(data->point.y, b_t.bottom.offset))
   {
      return b_t.bottom.normal;
   }
   if (closeEnough(data->point.y, b_t.top.offset))
   {
      return b_t.top.normal;
   }
   if (closeEnough(data->point.z, b_t.back.offset))
   {
      return b_t.back.normal;
   }
   if (closeEnough(data->point.z, b_t.front.offset))
   {
      return b_t.front.normal;
   }
   cout << "shouldn't be here." << endl;
   return vec3();
}

int plane_hit(const plane_t & p_t, const Ray & ray, float *t, HitData *data)
{
   float denominator = dot(ray.dir, (p_t.normal));
   if (denominator == 0.0)
   {
      return 0;
   }
   vec3 p = p_t.normal * p_t.offset;
   vec3 pMinusL = p - ray.point;
   float numerator = dot(pMinusL, (p_t.normal));
   *t = numerator / denominator;
   if (*t >= MIN_T && *t <= MAX_DIST)
   {
      data->hit = 1;
      data->point = ray.point + ray.dir * (*t);
      data->t = (*t);
      data->hitType = PLANE_HIT;
      if (p_t.f.reflection > 0.0)
      {
         data->reflect = new vec3();
         *data->reflect = mReflect(ray.dir, plane_normal(p_t));
      }
      else
      {
         data->reflect = NULL;
      }
      return 1;
   }
   return 0;
}

vec3 plane_normal(const plane_t & p_t)
{
   return p_t.normal;
}

int sphere_hit(const sphere_t & s_t, const Ray & ray, float *t, HitData *data)
{
   // Optimized algorithm courtesy of "Real-Time Rendering, Third Edition".
   vec3 l = s_t.location - ray.point;
   float s = dot(l, (ray.dir));
   float l2 = dot(l, (l));
   float r2 = s_t.radius * s_t.radius;
   if (s < MIN_T && l2 > r2)
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
   data->hitType = SPHERE_HIT;
   if (l2 < r2)
   {
      data->hit = -1;
      if (s_t.f.reflection > 0.0)
      {
         data->reflect = new vec3();
         *data->reflect = mReflect(ray.dir, sphere_normal(s_t, data));
      }
      else
      {
         data->reflect = NULL;
      }
      return -1;
   }
   else
   {
      data->hit = 1;
      if (s_t.f.reflection > 0.0)
      {
         data->reflect = new vec3();
         *data->reflect = mReflect(ray.dir, sphere_normal(s_t, data));
      }
      else
      {
         data->reflect = NULL;
      }
      return 1;
   }
}

vec3 sphere_normal(const sphere_t & s_t, HitData *data)
{
   vec3 n = data->point - s_t.location;
   n = normalize(n);
   return n;
}

int triangle_hit(const triangle_t & t_t, const Ray & ray, float *t, HitData *data)
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
      data->hitType = TRIANGLE_HIT;
      return 1;
   }
   return 0;
}

vec3 triangle_normal(const triangle_t & t_t)
{
   vec3 s1 = t_t.c2 - t_t.c1;
   vec3 s2 = t_t.c3 - t_t.c1;
   return cross(s1, s2);
}
