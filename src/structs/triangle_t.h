/**
 * A struct representing a triangle.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _TRIANGLE_T_H
#define _TRIANGLE_T_H

#include "structs/vector.h"
#include "structs/pigment_t.h"
#include "structs/finish_t.h"

struct triangle_t
{
   // The first corner of the triangle.
   vec3_t c1;

   // The second corner of the triangle.
   vec3_t c2;

   // The third corner of the triangle.
   vec3_t c3;

   pigment_t p;

   finish_t f;

};
#endif
