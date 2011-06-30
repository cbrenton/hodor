/**
 * A class to hold various data resulting from an intersection.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _HITDATA_H
#define _HITDATA_H

#include "structs/vector.h"

class Geometry;

struct HitData
{
   int hit;

   int hitIndex;

   vec3_t point;

   float t;

   Geometry *object;

   int hitType;

   int objIndex;

   int faceIndex;

   vec3_t *reflect;

   vec3_t *refract;

};
#endif
