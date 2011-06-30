/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _RAY_T_H
#define _RAY_T_H

#include "structs/vector.h"

struct ray_t
{
   // The origin of the ray.
   vec3_t point;

   // The direction of the ray.
   vec3_t dir;

};
#endif
