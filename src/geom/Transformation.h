/**
 * Holds a transformation matrix.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _TRANSFORMATION_H
#define _TRANSFORMATION_H

#include "structs/vector.h"

class Transformation
{
   public:
      Transformation();
      
      ~Transformation() {};
      
      void setScale(vec3_t scaleVec);
      
      void setRotation(float x, float y, float z);
      
      void setTranslation(float x, float y, float z);

      //Transform<float, 3, Affine> m;

};

#endif
