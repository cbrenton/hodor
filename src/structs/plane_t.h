/**
 * A struct representing a plane.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PLANE_T_H
#define _PLANE_T_H

#include <Eigen/Dense>
#include "structs/Pigment.h"
#include "structs/Finish.h"

using Eigen::Vector3f;

struct plane_t
{
   // The normal of the plane.
   Vector3f normal;

   // The plane offset ('d' in the plane's implicit equation).
   float offset;

   Pigment p;

   Finish f;

};
#endif
