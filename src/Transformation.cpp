/**
 * Holds a transformation matrix.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include "Transformation.h"

using namespace Eigen;

Transformation::Transformation() {
  m = Transform<float, 3, Affine>();
}

void Transformation::setScale(Vector3f scaleVec)
{
  m.scale(scaleVec);
}

void Transformation::setRotation(float x, float y, float z)
{
  ;
}

void Transformation::setTranslation(float x, float y, float z)
{
  ;
}
