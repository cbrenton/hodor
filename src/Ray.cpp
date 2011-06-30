/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Ray.h"

Ray::Ray(vec3_t _point, vec3_t _dir) :
   point(_point), dir(_dir)
{
}
