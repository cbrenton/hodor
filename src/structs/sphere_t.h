/**
 * A struct representing a sphere.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _SPHERE_T_H
#define _SPHERE_T_H

#include <Eigen/Dense>
using Eigen::Vector3f;

struct sphere_t
{
   // The radius of the sphere.
   float radius;

   // The location in world space of the sphere.
   Vector3f location;

};
#endif
