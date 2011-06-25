/**
 * A class to hold various data resulting from an intersection.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _HITDATA_H
#define _HITDATA_H

#include <Eigen/Dense>

using Eigen::Vector3f;

class Geometry;

struct HitData
{
   int hit;

   int hitIndex;

   Vector3f point;

   float t;

   Geometry *object;

   Vector3f *reflect;

   Vector3f *refract;

};
#endif
