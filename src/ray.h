/**
 * Ray struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _RAY_H
#define _RAY_H

#include "vector.h"
#include "triangle.h"
#include "hit_data.h"

struct ray
{
   vec3 pt;
   vec3 dir;
};

extern bool hit(ray *r, triangle *tri, float *t = NULL, hitData *data = NULL);
extern void normalize(ray *r);
extern void normalize(ray &r);
extern vec_t length(ray *r);
extern vec_t length(ray &r);
extern void debug(ray *r);
extern void debug(ray &r);

#endif
