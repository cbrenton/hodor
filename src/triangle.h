/**
 * Triangle struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include "vertex.h"
#include "material.h"

struct triangle
{
   vertex *pts[3];
   material *mat;
};

inline void initTri(triangle *t)
{
   t->pts[0] = new vertex;
   initVert(t->pts[0]);
   t->pts[1] = new vertex;
   initVert(t->pts[1]);
   t->pts[2] = new vertex;
   initVert(t->pts[2]);
}

#endif
