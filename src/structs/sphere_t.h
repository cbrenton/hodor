/**
 * A struct representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _SPHERE_T_H
#define _SPHERE_T_H

#include <glm/glm.hpp>
#include "structs/Pigment.h"
#include "structs/Finish.h"

struct sphere_t
{
   // The radius of the sphere.
   float radius;

   // The location in world space of the sphere.
   glm::vec3 location;

   Pigment p;

   Finish f;

};
#endif
