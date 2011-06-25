/**
 * A struct representing a box.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _BOX_T_H
#define _BOX_T_H

#include <Eigen/Dense>
using Eigen::Vector3f;
#include "structs/plane_t.h"

struct box_t
{
   // The first corner of the box.
   Vector3f c1;

   // The second corner of the box.
   Vector3f c2;

   plane_t left;

   plane_t right;

   plane_t bottom;

   plane_t top;

   plane_t back;

   plane_t front;

};
#endif
