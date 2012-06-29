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
   return Box();
}

int Geometry::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   return 0;
}

vec3 Geometry::getNormal(const vec3 & point)
{
   return vec3();
}

/*
void Geometry::addTransformation(Transform<float, 3, Affine> t)
{
}
*/
