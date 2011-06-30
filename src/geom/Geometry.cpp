/**
 * The abstract class representing a geometry object in a scene. Hit detection per object is done here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Geometry.h"
#include "Ray.h"
#include "Box.h"
#include "structs/HitData.h"

Box Geometry::bBox()
{
   //return Box();
   Box result;
   return result;
}

int Geometry::hit(Ray & ray, float *t, HitData *data, float minT, float maxT)
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
