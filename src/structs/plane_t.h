/**
 * A struct representing a plane.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PLANE_T_H
#define _PLANE_T_H

#include "structs/vector.h"
#include "structs/Pigment.h"
#include "structs/Finish.h"

struct plane_t
{
   // The normal of the plane.
   vec3_t normal;

   // The plane offset ('d' in the plane's implicit equation).
   float offset;

   Pigment p;

   Finish f;

};
#endif
