/**
 * Global utilities needed in all sorts of classes, as well as constants.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _GLOBALS_H
#define _GLOBALS_H

#include <cstdlib>
#include <stdint.h>

#define EPSILON 0.001f
#define MIN_T 0.0f
#define MAX_DIST 10000.0f
#define BOX_HIT 0
#define MESH2_HIT 1
#define PLANE_HIT 2
#define SPHERE_HIT 3
#define TRIANGLE_HIT 4
#define COLOR_T uint16_t
#define COLOR_RANGE 65536.0

#define mPR_VEC(a) printf("<%f, %f, %f>", (a).x, (a).y, (a).z)
#define mPRLN_VEC(a) printf("<%f, %f, %f>\n", (a).x, (a).y, (a).z)

inline int randInt()
{
   return rand();
}

inline float randFloat()
{
   return (float)rand() / (float)RAND_MAX;
}

/*
inline float max3(float a, float b, float c)
{
   return std::max(std::max(a, b), c);
}

inline float min3(float a, float b, float c)
{
   return std::min(std::min(a, b), c);
}
*/

inline bool closeEnough(float a, float b)
{
   return a - b <= EPSILON || b - a <= EPSILON;
}

#endif
