/**
 * Triangle vertex.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _VERTEX_H
#define _VERTEX_H

#include <stdio.h>
#include "glm/glm.hpp"

struct vertex
{
   glm::vec3 coord;
   glm::vec3 texCoord;
   glm::vec3 normal;
};

inline void debug(vertex *v)
{
   printf("vertex: ");
   //debug(v->coord);
}

#endif
