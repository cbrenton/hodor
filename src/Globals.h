/**
 * Global utilities needed in all sorts of classes, as well as constants.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include <cstdlib>

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
