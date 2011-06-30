/**
 * A struct representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _SPHERE_T_H
#define _SPHERE_T_H

#include "structs/vector.h"
#include "structs/pigment_t.h"
#include "structs/finish_t.h"

struct sphere_t
{
   // The radius of the sphere.
   float radius;

   // The location in world space of the sphere.
   vec3_t location;

   pigment_t p;

   finish_t f;

};
#endif
