/**
 * Operations for the ray struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#include "ray.h"

bool hit(ray *r, triangle *tri, float *t, hitData *data)
{
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
