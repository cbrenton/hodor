/**
 * A class to hold various data resulting from an intersection.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _HITD_T_H
#define _HITD_T_H

#include "structs/cuda_vector.h"

struct hitd_t
{
   int hit;

   int objIndex;

   vec3d_t point;

   float t;

   int hitType;

};
#endif
