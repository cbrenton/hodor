/**
 * A struct representing a triangle.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _TRIANGLE_T_H
#define _TRIANGLE_T_H

#include <glm/glm.hpp>
#include "structs/Pigment.h"
#include "structs/Finish.h"

struct triangle_t
{
   // The first corner of the triangle.
   glm::vec3 c1;

   // The second corner of the triangle.
   glm::vec3 c2;

   // The third corner of the triangle.
   glm::vec3 c3;

   Pigment p;

   Finish f;

};
#endif
