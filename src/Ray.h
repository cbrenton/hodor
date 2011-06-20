/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _RAY_H
#define _RAY_H

class Ray
{
  public:
    //The origin of the ray.
    Vector3f point;

    //The direction of the ray.
    Vector3f dir;

};
#endif
