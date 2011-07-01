/**
 * Contains intersection data for all geometry structs.
 * @author Chris Brenton
 * @date 6/24/2011
 */

#include "cuPrintf.cu"
#include "hit_kernel.h"
#include "Globals.h"
#include "structs/vector.h"

int box_hit(box_t *b_t, ray_t & ray, float *t, hit_t *data)
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
   data->point = ray.dir * (*t);
   data->point += ray.point;
   data->t = (*t);
   data->hitType = BOX_HIT;
   if (b_t->f.reflection > 0.0)
   {
      data->reflect = new vec3_t();
      vec3_t n = box_normal(b_t, data);
      *data->reflect = mReflect(ray.dir, n);
   }
   else
   {
      data->reflect = NULL;
   }
   return 1;
}

vec3_t box_normal(box_t *b_t, hit_t *data)
{
   if (closeEnough(data->point.x(), b_t->left.offset))
   {
      return b_t->left.normal;
   }
   if (closeEnough(data->point.x(), b_t->right.offset))
   {
      return b_t->right.normal;
   }
   if (closeEnough(data->point.y(), b_t->bottom.offset))
   {
      return b_t->bottom.normal;
   }
   if (closeEnough(data->point.y(), b_t->top.offset))
   {
      return b_t->top.normal;
   }
   if (closeEnough(data->point.z(), b_t->back.offset))
   {
      return b_t->back.normal;
   }
   if (closeEnough(data->point.z(), b_t->front.offset))
   {
      return b_t->front.normal;
   }
   cout << "shouldn't be here." << endl;
   return vec3_t();
}

int plane_hit(plane_t *p_t, ray_t & ray, float *t, hit_t *data)
{
   float denominator = ray.dir.dot(p_t->normal);
   if (denominator == 0.0)
   {
      return 0;
   }
   vec3_t p = p_t->normal * p_t->offset;
   vec3_t pMinusL = p - ray.point;
   float numerator = pMinusL.dot(p_t->normal);
   *t = numerator / denominator;
   if (*t >= MIN_T && *t <= MAX_DIST)
   {
      data->hit = 1;
      data->point = ray.dir * (*t);
      data->point += ray.point;
      data->t = (*t);
      data->hitType = PLANE_HIT;
      if (p_t->f.reflection > 0.0)
      {
         data->reflect = new vec3_t();
         vec3_t n = plane_normal(p_t);
         *data->reflect = mReflect(ray.dir, n);
      }
      else
      {
         data->reflect = NULL;
      }
      return 1;
   }
   return 0;
}

vec3_t plane_normal(plane_t *p_t)
{
   return p_t->normal;
}

int sphere_hit(sphere_t & s_t, ray_t & ray, float *t, hit_t *data)
{
   // Optimized algorithm courtesy of "Real-Time Rendering, Third Edition".
   vec3_t l = s_t.location - ray.point;
   float s = l.dot(ray.dir);
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
   data->point = ray.dir * (*t);
   data->point += ray.point;
   data->hitType = SPHERE_HIT;
   if (l2 < r2)
   {
      data->hit = -1;
      if (s_t.f.reflection > 0.0)
      {
         data->reflect = new vec3_t();
         vec3_t n = sphere_normal(s_t, data);
         *data->reflect = mReflect(ray.dir, n);
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
         data->reflect = new vec3_t();
         vec3_t n = sphere_normal(s_t, data);
         *data->reflect = mReflect(ray.dir, n);
      }
      else
      {
         data->reflect = NULL;
      }
      return 1;
   }
}

vec3_t sphere_normal(sphere_t & s_t, hit_t *data)
{
   vec3_t n = data->point - s_t.location;
   n.normalize();
   return n;
}

int triangle_hit(triangle_t *t_t, ray_t & ray, float *t, hit_t *data)
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
      data->point = ray.dir * (*t);
      data->point += ray.point;
      data->t = (*t);
      data->hitType = TRIANGLE_HIT;
      return 1;
   }
   return 0;
}

vec3_t triangle_normal(triangle_t *t_t)
{
   vec3_t s1 = t_t->c2 - t_t->c1;
   vec3_t s2 = t_t->c3 - t_t->c1;
   s1.cross(s1, s2);
   return s1;
}

__global__ void hitSpheres(sphere_t *spheres, int sphere_size, ray_t **rays, hit_t *results)
{
   /*
      int thread_id = (blockIdx.y * BLOCKS_PER_ROW * THREADS_PER_BLOCK) +
      blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

      cuPrintf("%d\n", thread_id);
    */

   /*
   // INITIALIZE closestT to MAX_DIST + 0.1
   float closestT = MAX_DIST + 0.1f;
   // INITIALIZE closestData to empty hit_t
   hit_t *closestData = results[thread_id];

   // FOR each item in spheres
   for (int sphereNdx = 0; sphereNdx < (int)spheres.size(); sphereNdx++)
   {
   float sphereT = -1;
   hit_t *sphereData = new hit_t();
   // IF current item is hit by ray
   if (sphere_hit(spheres[sphereNdx], ray, &sphereT, sphereData) != 0)
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
   delete sphereData;
   }
    */
}

//__global__ void cuda_test(ray_t **rays, int width, int height, sphere_t *spheres, int sphere_size, hit_t* results)
__global__ void cuda_test(ray_t **rays, int width, int height, sphere_t *spheres, int sphere_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   int x = (idx % width);
   int y = idx / width;
   if (idx >= width * height) return;
   if (x == 0)
   {
      cuPrintf("id %d: (%d, %d)\n", idx, x, y);
   }
   //results[idx].hit = x;
}

void initPrintf()
{
   cudaPrintfInit();
}
   
void endPrintf()
{
   cudaPrintfDisplay(stdout, true);
   cudaPrintfEnd();
}
