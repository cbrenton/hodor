/**
 * A struct representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _SPHERE_T_H
#define _SPHERE_T_H

#include <Eigen/Dense>
#include "structs/Pigment.h"
#include "structs/Finish.h"

using Eigen::Vector3f;

struct sphere_t
{
   // The radius of the sphere.
   float radius;

   // The location in world space of the sphere.
   Vector3f location;

   Pigment p;

   Finish f;

};
#endif
