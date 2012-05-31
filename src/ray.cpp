/**
 * Operations for the ray struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#include "ray.h"

bool hit(ray *ray_in, triangle *tri, vec_t *t, hit_data *data)
{
   float result = -1;
   vec_t bBeta, bGamma, bT;

   mat3 A =
   {{
      tri->pts[0]->coord.v[0]-tri->pts[1]->coord.v[0],
      tri->pts[0]->coord.v[0]-tri->pts[2]->coord.v[0],
      ray_in->dir.v[0],
      tri->pts[0]->coord.v[1]-tri->pts[1]->coord.v[1],
      tri->pts[0]->coord.v[1]-tri->pts[2]->coord.v[1],
      ray_in->dir.v[1],
      tri->pts[0]->coord.v[2]-tri->pts[1]->coord.v[2],
      tri->pts[0]->coord.v[2]-tri->pts[2]->coord.v[2],
      ray_in->dir.v[2]
   }};
   vec_t detA = det3(A);

   mat3 baryT =
   {{
      tri->pts[0]->coord.v[0]-tri->pts[1]->coord.v[0],
      tri->pts[0]->coord.v[0]-tri->pts[2]->coord.v[0],
      tri->pts[0]->coord.v[0]-ray_in->pt.v[0],
      tri->pts[0]->coord.v[1]-tri->pts[1]->coord.v[1],
      tri->pts[0]->coord.v[1]-tri->pts[2]->coord.v[1],
      tri->pts[0]->coord.v[1]-ray_in->pt.v[1],
      tri->pts[0]->coord.v[2]-tri->pts[1]->coord.v[2],
      tri->pts[0]->coord.v[2]-tri->pts[2]->coord.v[2],
      tri->pts[0]->coord.v[2]-ray_in->pt.v[2]
   }};

   bT = det3(baryT) / detA;

   if (bT < 0)
   {
      result = 0;
   }
   else
   {
      mat3 baryGamma =
      {{
         tri->pts[0]->coord.v[0]-tri->pts[1]->coord.v[0],
         tri->pts[0]->coord.v[0]-ray_in->pt.v[0],
         ray_in->dir.v[0],
         tri->pts[0]->coord.v[1]-tri->pts[1]->coord.v[1],
         tri->pts[0]->coord.v[1]-ray_in->pt.v[1],
         ray_in->dir.v[1],
         tri->pts[0]->coord.v[2]-tri->pts[1]->coord.v[2],
         tri->pts[0]->coord.v[2]-ray_in->pt.v[2],
         ray_in->dir.v[2]
       }};

      bGamma = det3(baryGamma) / detA;

      if (bGamma < 0 || bGamma > 1)
      {
         result = 0;
      }
      else
      {
         mat3 baryBeta =
         {{
            tri->pts[0]->coord.v[0]-ray_in->pt.v[0],
            tri->pts[0]->coord.v[0]-tri->pts[2]->coord.v[0],
            ray_in->dir.v[0],
            tri->pts[0]->coord.v[1]-ray_in->pt.v[1],
            tri->pts[0]->coord.v[1]-tri->pts[2]->coord.v[1],
            ray_in->dir.v[1],
            tri->pts[0]->coord.v[2]-ray_in->pt.v[2],
            tri->pts[0]->coord.v[2]-tri->pts[2]->coord.v[2],
            ray_in->dir.v[2]
         }};

         bBeta = det3(baryBeta) / detA;

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
      data->pt = ray_in->pt;
      data->pt = ray_in->dir;
      multiply(data->pt, *t);
      add(data->pt, ray_in->dir);
      data->t = (*t);
      return true;
   }
   return false;
}

void normalize(ray *r)
{
   vec_t len = length(r);

   subtract(r->dir, r->pt);
   multiply(r->dir, 1.f / len);
}

void normalize(ray &r)
{
   vec_t len = length(r);

   subtract(r.dir, r.pt);
   multiply(r.dir, 1.f / len);
}

vec_t length(ray *r)
{
   return distance3(r->dir, r->pt);
}

vec_t length(ray &r)
{
   return distance3(r.dir, r.pt);
}

void debug(ray *r)
{
   printf("pt: ");
   debug(r->pt);
   printf("dir: ");
   debug(r->dir);
}

void debug(ray &r)
{
   printf("pt: ");
   debug(r.pt);
   printf("dir: ");
   debug(r.dir);
}
