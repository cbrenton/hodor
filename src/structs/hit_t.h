/**
 * A class to hold various data resulting from an intersection.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _HIT_T_H
#define _HIT_T_H

#include "structs/vector.h"

class Geometry;

struct hit_t
{
   int hit;

   int hitIndex;

   vec3_t point;

   float t;

   Geometry *object;

   int hitType;

   int objIndex;

   int faceIndex;

   //vec3_t *reflect;

   //vec3_t *refract;

};
#endif
