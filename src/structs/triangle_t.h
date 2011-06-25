/**
 * A struct representing a triangle.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _TRIANGLE_T_H
#define _TRIANGLE_T_H

#include <Eigen/Dense>
#include "structs/Pigment.h"
#include "structs/Finish.h"

using Eigen::Vector3f;

struct triangle_t
{
   // The first corner of the triangle.
   Vector3f c1;

   // The second corner of the triangle.
   Vector3f c2;

   // The third corner of the triangle.
   Vector3f c3;

   Pigment p;

   Finish f;

};
#endif
