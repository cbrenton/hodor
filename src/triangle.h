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

#endif
