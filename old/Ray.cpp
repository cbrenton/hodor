/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Ray.h"

Ray::Ray(Eigen::Vector3f _point, Eigen::Vector3f _dir) :
   point(_point), dir(_dir)
{
}
