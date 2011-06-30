/**
 * The abstract class representing a geometry object in a scene. Hit detection per object is done here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Geometry.h"
#include "structs/ray_t.h"
#include "Box.h"
#include "structs/hit_t.h"

Box Geometry::bBox()
{
   //return Box();
   Box result;
   return result;
}

int Geometry::hit(ray_t & ray, float *t, hit_t *data, float minT, float maxT)
{
   return 0;
}

vec3_t Geometry::getNormal(vec3_t & point)
{
   return vec3_t();
}

/*
void Geometry::addTransformation(Transform<float, 3, Affine> t)
{
}
*/
