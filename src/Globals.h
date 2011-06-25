/**
 * Global utilities needed in all sorts of classes, as well as constants.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _GLOBALS_H
#define _GLOBALS_H

#include <cstdlib>
#include <Eigen/Dense>


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

/*
Eigen::Vector3f reflect(Eigen::Vector3f d, Eigen::Vector3f n)
{
   return n * (2 * (-d.dot(n))) + d;
}
*/

#define mReflect(d, n) ((n) * (2 * (-(d).dot(n))) + d)

#define mPrintVec(v) cout << "<" << v.x() << ", " << v.y() << ", " << v.z() << ">" << endl;

#endif
