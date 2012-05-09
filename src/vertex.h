/**
 * Triangle vertex.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _VERTEX_H
#define _VERTEX_H

#include <stdio.h>
#include "vector.h"

struct vertex
{
   vec3 coord;
   vec2 texCoord;
   vec3 normal;
};

inline void debug(vertex *v)
{
   debug(v->coord);
}

#endif
