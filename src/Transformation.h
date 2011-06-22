/**
 * Holds a transformation matrix.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _TRANSFORMATION_H
#define _TRANSFORMATION_H

#include <Eigen/Dense>

class Transformation
{
  public:
    Transformation();
    ~Transformation() {};
    void setScale(Eigen::Vector3f scaleVec);
    void setRotation(float x, float y, float z);
    void setTranslation(float x, float y, float z);
    Eigen::Transform<float, 3, Eigen::Affine> m;

};

#endif
