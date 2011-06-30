/**
 * Global utilities needed in all sorts of classes, as well as constants.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _GLOBALS_H
#define _GLOBALS_H

#include <cstdlib>
#include <cmath>
#include "structs/box_t.h"
#include "structs/plane_t.h"
#include "structs/sphere_t.h"
#include "structs/triangle_t.h"
#include "structs/hit_t.h"
#include "structs/ray_t.h"

#define EPSILON 0.001f
#define MIN_T 0.0f
#define MAX_DIST 10000.0f
#define BOX_HIT 0
#define MESH2_HIT 1
#define PLANE_HIT 2
#define SPHERE_HIT 3
#define TRIANGLE_HIT 4

inline int randInt()
{
   return rand();
}

inline float randFloat()
{
   return (float)rand() / (float)RAND_MAX;
}

inline float max3(float a, float b, float c)
{
   return std::max(std::max(a, b), c);
}

inline float min3(float a, float b, float c)
{
   return std::min(std::min(a, b), c);
}

inline bool closeEnough(float a, float b)
{
   return abs(a - b) <= EPSILON;
}

#define mReflect(d, n) ((n) * (2 * (-(d).dot(n))) + d)

#define mPrintVec(v) cout << "<" << v.x() << ", " << v.y() << ", " << v.z() << ">" << endl;

#endif
