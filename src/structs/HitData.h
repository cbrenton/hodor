/**
 * A class to hold various data resulting from an intersection.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _HITDATA_H
#define _HITDATA_H

#include <glm/glm.hpp>

class Geometry;

struct HitData
{
   int hit;

   int hitIndex;

   glm::vec3 point;

   float t;

   Geometry *object;

   int hitType;

   int objIndex;

   int faceIndex;

   glm::vec3 *reflect;

   glm::vec3 *refract;

};
#endif
