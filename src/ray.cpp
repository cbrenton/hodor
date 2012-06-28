/**
 * Operations for the ray struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#include "ray.h"

using namespace glm;

bool hit(ray *ray_in, triangle *tri, float *t, hit_data *data)
{
   float result = -1;
   float bBeta, bGamma, bT;

   mat3 A (
         tri->pts[0]->coord.x-tri->pts[1]->coord.x,
         tri->pts[0]->coord.x-tri->pts[2]->coord.x,
         ray_in->dir.x,
         tri->pts[0]->coord.y-tri->pts[1]->coord.y,
         tri->pts[0]->coord.y-tri->pts[2]->coord.y,
         ray_in->dir.y,
         tri->pts[0]->coord.z-tri->pts[1]->coord.z,
         tri->pts[0]->coord.z-tri->pts[2]->coord.z,
         ray_in->dir.z
         );
   float detA = determinant(A);

   mat3 baryT (
         tri->pts[0]->coord.x-tri->pts[1]->coord.x,
         tri->pts[0]->coord.x-tri->pts[2]->coord.x,
         tri->pts[0]->coord.x-ray_in->pt.x,
         tri->pts[0]->coord.y-tri->pts[1]->coord.y,
         tri->pts[0]->coord.y-tri->pts[2]->coord.y,
         tri->pts[0]->coord.y-ray_in->pt.y,
         tri->pts[0]->coord.z-tri->pts[1]->coord.z,
         tri->pts[0]->coord.z-tri->pts[2]->coord.z,
         tri->pts[0]->coord.z-ray_in->pt.z
         );

   bT = determinant(baryT) / detA;

   if (bT < 0)
   {
      //printf("break at bT\n");
      result = 0;
   }
   else
   {
      mat3 baryGamma (
            tri->pts[0]->coord.x-tri->pts[1]->coord.x,
            tri->pts[0]->coord.x-ray_in->pt.x,
            ray_in->dir.x,
            tri->pts[0]->coord.y-tri->pts[1]->coord.y,
            tri->pts[0]->coord.y-ray_in->pt.y,
            ray_in->dir.y,
            tri->pts[0]->coord.z-tri->pts[1]->coord.z,
            tri->pts[0]->coord.z-ray_in->pt.z,
            ray_in->dir.z
            );

      bGamma = determinant(baryGamma) / detA;

      if (bGamma < 0 || bGamma > 1)
      {
         //debug(tri);
         //printf("break at bGamma\n");
         /*
         if (bGamma < 0)
         {
            printf("\tbGamma < 0\n");
         }
         else
         {
            printf("\tbGamma > 1\n");
         }
         */
         result = 0;
      }
      else
      {
         mat3 baryBeta (
               tri->pts[0]->coord.x-ray_in->pt.x,
               tri->pts[0]->coord.x-tri->pts[2]->coord.x,
               ray_in->dir.x,
               tri->pts[0]->coord.y-ray_in->pt.y,
               tri->pts[0]->coord.y-tri->pts[2]->coord.y,
               ray_in->dir.y,
               tri->pts[0]->coord.z-ray_in->pt.z,
               tri->pts[0]->coord.z-tri->pts[2]->coord.z,
               ray_in->dir.z
               );

         bBeta = determinant(baryBeta) / detA;

         if (bBeta < 0 || bBeta > 1 - bGamma)
         {
            //printf("break at bBeta\n");
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
      data->beta = bBeta;
      data->gamma = bGamma;
      data->alpha = 1 - bBeta - bGamma;
      data->pt = ray_in->pt;
      data->pt = ray_in->dir;
      data->pt *= *t;
      data->pt += ray_in->dir;
      data->t = *t;
      return true;
   }
   return false;
}

void normalize(ray *r)
{
   float len = length(r);

   r->dir -= r->pt;
   r->dir *= 1.f / len;
}

void normalize(ray &r)
{
   float len = length(r);

   r.dir -= r.pt;
   r.dir *= 1.f / len;
}

float length(ray *r)
{
   return distance(r->dir, r->pt);
}

float length(ray &r)
{
   return distance(r.dir, r.pt);
}

void debug(ray *r)
{
   printf("pt: ");
   mPRLN_VEC(r->pt);
   printf("dir: ");
   mPRLN_VEC(r->dir);
}

void debug(ray &r)
{
   printf("pt: ");
   mPRLN_VEC(r.pt);
   printf("dir: ");
   mPRLN_VEC(r.dir);
}
