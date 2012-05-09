/**
 * Ray struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _RAY_H
#define _RAY_H

#include "globals.h"
#include "vector.h"
#include "triangle.h"
#include "hit_data.h"

struct ray
{
   vec3 pt;
   vec3 dir;
};

extern bool hit(ray *ray_in, triangle *tri, vec_t *t = NULL, hit_data *data = NULL);
extern void normalize(ray *r);
extern void normalize(ray &r);
extern vec_t length(ray *r);
extern vec_t length(ray &r);
extern void debug(ray *r);
extern void debug(ray &r);

#endif
