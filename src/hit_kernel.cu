/**
 * Contains intersection data for all geometry structs.
 * @author Chris Brenton
 * @date 6/24/2011
 */

#include <cutil_math.h>
#include "hit_kernel.h"
#include "Globals.h"
#include "structs/vector.h"
#include "structs/hitd_t.h"
#include "structs/hit_t.h"

int cpu_hit(box_t *b_t, ray_t & ray, float *t, hitd_t *data)
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
      float xL = min(b_t->c1[i], b_t->c2[i]);
      // Component of the maxinum plane location in the current dimension.
      float xH = max(b_t->c1[i], b_t->c2[i]);
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
   data->t = (*t);
   data->hitType = BOX_HIT;
   /*
      if (b_t->f.reflection > 0.0)
      {
      data->reflect = new vec3d_t();
      vec3d_t n = box_normal(b_t, data);
    *data->reflect = mReflect(ray.dir, n);
    }
    else
    {
    data->reflect = NULL;
    }
    */
   return 1;
}

int cpu_hit(plane_t *p_t, ray_t & ray, float *t, hitd_t *data)
{
   /*
      float denominator = ray.dir.dot(p_t->normal);
      if (denominator == 0.0)
      {
      return 0;
      }
      vec3d_t p = p_t->normal * p_t->offset;
      vec3d_t pMinusL = p - ray.point;
      float numerator = pMinusL.dot(p_t->normal);
    *t = numerator / denominator;
    if (*t >= MIN_T && *t <= MAX_DIST)
    {
    data->hit = 1;
    data->t = (*t);
    data->hitType = PLANE_HIT;
    */

   /*
      if (p_t->f.reflection > 0.0)
      {
      data->reflect = new vec3d_t();
      vec3d_t n = plane_normal(p_t);
    *data->reflect = mReflect(ray.dir, n);
    }
    else
    {
    data->reflect = NULL;
    }
    */

   //return 1;
   //}
   return 0;
}

int cpu_hit(sphere_t & s_t, ray_t & ray, float *t, hit_t *data)
{
   vec3_t location = s_t.location;
   vec3_t rayPoint = ray.point;
   vec3_t l = location - rayPoint;
   vec3_t rayDir = ray.dir;
   float s = l.dot(rayDir);
   float l2 = l.dot(l);
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
   data->hitType = SPHERE_HIT;
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

int cpu_hit(triangle_t *t_t, ray_t & ray, float *t, hitd_t *data)
{
   float result = -1;
   float bBeta, bGamma, bT;

   float A[9];
   A[0] = t_t->c1.x()-t_t->c2.x();
   A[1] = t_t->c1.x()-t_t->c3.x();
   A[2] = ray.dir.x();
   A[3] = t_t->c1.y()-t_t->c2.y();
   A[4] = t_t->c1.y()-t_t->c3.y();
   A[5] = ray.dir.y();
   A[6] = t_t->c1.z()-t_t->c2.z();
   A[7] = t_t->c1.z()-t_t->c3.z();
   A[8] = ray.dir.z();
   float detA = (A[0]*A[4]*A[8]) + (A[1]*A[5]*A[6]) + (A[2]*A[3]*A[7]) -
      (A[0]*A[5]*A[7]) - (A[1]*A[3]*A[8]) - (A[2]*A[4]*A[6]);

   float baryT[9];
   baryT[0] = t_t->c1.x()-t_t->c2.x();
   baryT[1] = t_t->c1.x()-t_t->c3.x();
   baryT[2] = t_t->c1.x()-ray.point.x();

   baryT[3] = t_t->c1.y()-t_t->c2.y();
   baryT[4] = t_t->c1.y()-t_t->c3.y();
   baryT[5] = t_t->c1.y()-ray.point.y();

   baryT[6] = t_t->c1.z()-t_t->c2.z();
   baryT[7] = t_t->c1.z()-t_t->c3.z();
   baryT[8] = t_t->c1.z()-ray.point.z();

   float detBaryT = (baryT[0]*baryT[4]*baryT[8]) +
      (baryT[1]*baryT[5]*baryT[6]) +
      (baryT[2]*baryT[3]*baryT[7]) -
      (baryT[0]*baryT[5]*baryT[7]) -
      (baryT[1]*baryT[3]*baryT[8]) -
      (baryT[2]*baryT[4]*baryT[6]);

   bT = detBaryT / detA;

   if (bT < 0)
   {
      result = 0;
   }
   else
   {
      float baryGamma[9];
      baryGamma[0] = t_t->c1.x()-t_t->c2.x();
      baryGamma[1] = t_t->c1.x()-ray.point.x();
      baryGamma[2] = ray.dir.x();

      baryGamma[3] = t_t->c1.y()-t_t->c2.y();
      baryGamma[4] = t_t->c1.y()-ray.point.y();
      baryGamma[5] = ray.dir.y();

      baryGamma[6] = t_t->c1.z()-t_t->c2.z();
      baryGamma[7] = t_t->c1.z()-ray.point.z();
      baryGamma[8] = ray.dir.z();

      float detBaryGamma = (baryGamma[0]*baryGamma[4]*baryGamma[8]) +
         (baryGamma[1]*baryGamma[5]*baryGamma[6]) +
         (baryGamma[2]*baryGamma[3]*baryGamma[7]) -
         (baryGamma[0]*baryGamma[5]*baryGamma[7]) -
         (baryGamma[1]*baryGamma[3]*baryGamma[8]) -
         (baryGamma[2]*baryGamma[4]*baryGamma[6]);

      bGamma = detBaryGamma / detA;

      if (bGamma < 0 || bGamma > 1)
      {
         result = 0;
      }
      else
      {
         float baryBeta[9];
         baryBeta[0] = t_t->c1.x()-ray.point.x();
         baryBeta[1] = t_t->c1.x()-t_t->c3.x();
         baryBeta[2] = ray.dir.x();

         baryBeta[3] = t_t->c1.y()-ray.point.y();
         baryBeta[4] = t_t->c1.y()-t_t->c3.y();
         baryBeta[5] = ray.dir.y();

         baryBeta[6] = t_t->c1.z()-ray.point.z();
         baryBeta[7] = t_t->c1.z()-t_t->c3.z();
         baryBeta[8] = ray.dir.z();

         float detBaryBeta = (baryBeta[0]*baryBeta[4]*baryBeta[8]) +
            (baryBeta[1]*baryBeta[5]*baryBeta[6]) +
            (baryBeta[2]*baryBeta[3]*baryBeta[7]) -
            (baryBeta[0]*baryBeta[5]*baryBeta[7]) -
            (baryBeta[1]*baryBeta[3]*baryBeta[8]) -
            (baryBeta[2]*baryBeta[4]*baryBeta[6]);

         bBeta = detBaryBeta / detA;

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
      data->t = (*t);
      data->hitType = TRIANGLE_HIT;
      return 1;
   }
   return 0;
}

__device__ int sphere_hit(sphere_t & s_t, ray_t & ray, float *t, hitd_t *data)
{
   // Optimized algorithm courtesy of "Real-Time Rendering, Third Edition".
   vec3d_t location = s_t.location;
   vec3d_t rayPoint = ray.point;
   vec3d_t l = location - rayPoint;
   vec3d_t rayDir = ray.dir;
   float s = l.dot(rayDir);
   float l2 = l.dot(l);
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
   data->hitType = SPHERE_HIT;
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

__device__ int plane_hit(plane_t & p_t, ray_t & ray, float *t, hitd_t *data)
{
   vec3d_t rayDir = ray.dir;
   vec3d_t normal = p_t.normal;
   float denominator = rayDir.dot(normal);
   if (denominator == 0.0)
   {
      return 0;
   }
   vec3d_t p = normal * p_t.offset;
   //vec3d_t pMinusL = p - ray.point;
   vec3d_t pMinusL = ray.point;
   pMinusL *= -1.0f;
   pMinusL += p;
   float numerator = pMinusL.dot(normal);
   *t = numerator / denominator;
   if (*t >= MIN_T && *t <= MAX_DIST)
   {
      data->hit = 1;
      data->t = (*t);
      data->hitType = PLANE_HIT;
      return 1;
   }
   return 0;
}

vec3_t normal(box_t *b_t, hitd_t & data)
{
   /*
      if (closeEnough(data.point.x(), b_t->left.offset))
      {
      return vec3d_t(b_t->left.normal);
      }
      if (closeEnough(data.point.x(), b_t->right.offset))
      {
      return vec3d_t(b_t->right.normal);
      }
      if (closeEnough(data.point.y(), b_t->bottom.offset))
      {
      return vec3d_t(b_t->bottom.normal);
      }
      if (closeEnough(data.point.y(), b_t->top.offset))
      {
      return vec3d_t(b_t->top.normal);
      }
      if (closeEnough(data.point.z(), b_t->back.offset))
      {
      return vec3d_t(b_t->back.normal);
      }
      if (closeEnough(data.point.z(), b_t->front.offset))
      {
      return vec3d_t(b_t->front.normal);
      }
      cout << "shouldn't be here." << endl;
    */
   return vec3_t();
}

vec3_t normal(plane_t & p_t)
{
   return p_t.normal;
}

vec3_t normal(sphere_t & s_t, vec3_t & data)
{
   vec3_t n = data - s_t.location;
   n.normalize();
   return n;
}

vec3_t normal(triangle_t *t_t)
{
   /*
      vec3d_t s1 = t_t->c2 - t_t->c1;
      vec3d_t s2 = t_t->c3 - t_t->c1;
      s1.cross(s1, s2);
      return s1;
    */
   return vec3_t(0, 0, 0);
}

__global__ void set_spheres(sphere_t *spheresIn, int numSpheres)
{
   spheres = spheresIn;
   spheres_size = numSpheres;
}

__global__ void set_planes(plane_t *planesIn, int numplanes)
{
   planes = planesIn;
   planes_size = numplanes;
}

//__global__ void cuda_hit(ray_t *rays, int num, sphere_t *spheres,
//int spheres_size, hitd_t *results)
__global__ void cuda_hit(ray_t *rays, int num, hitd_t *results)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx >= num)
   {
      return;
   }

   // INITIALIZE closestT to MAX_DIST + 0.1
   float closestT = MAX_DIST + 0.1f;
   // INITIALIZE closestData to empty hitd_t
   hitd_t *closestData = &results[idx];

   //ray_t *ray = &rays[num];
   ray_t *ray = &rays[idx];

   hitd_t *sphereData = new hitd_t();
   // FOR each item in spheres
   for (int sphereNdx = 0; sphereNdx < (int)spheres_size; sphereNdx++)
   {
      float sphereT = -1;
      //// IF current item is hit by ray
      if (sphere_hit(spheres[sphereNdx], *ray, &sphereT, sphereData) != 0)
      {
         // IF intersection is closer than closestT
         if (sphereT < closestT)
         {
            // SET closestT to intersection
            closestT = sphereT;
            // SET closestData to intersection data
            *closestData = *sphereData;
            closestData->objIndex = sphereNdx;
         }
         // ENDIF
      }
      // ENDIF
   }
   delete sphereData;

   hitd_t *planeData = new hitd_t();
   // FOR each item in planes
   for (int planeNdx = 0; planeNdx < (int)planes_size; planeNdx++)
   {
      float planeT = -1;
      //// IF current item is hit by ray
      if (plane_hit(planes[planeNdx], *ray, &planeT, planeData) != 0)
      {
         // IF intersection is closer than closestT
         if (planeT < closestT)
         {
            // SET closestT to intersection
            closestT = planeT;
            // SET closestData to intersection data
            *closestData = *planeData;
            closestData->objIndex = planeNdx;
         }
         // ENDIF
      }
      // ENDIF
   }
   delete planeData;

}
