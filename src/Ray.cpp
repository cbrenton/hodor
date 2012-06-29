/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Ray.h"

Ray::Ray(glm::vec3 _point, glm::vec3 _dir) :
   point(_point), dir(_dir)
{
}
