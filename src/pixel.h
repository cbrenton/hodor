/**
 * Holds color data for a single pixel.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PIXEL_H
#define _PIXEL_H

#include "structs/color.h"
#include "glm/glm.hpp"

class Pixel
{
   public:
      Pixel() {};

      Pixel(int r, int g, int b);
      
      Pixel(float r, float g, float b);

      void clamp();

      void add(Pixel other);

      void multiply(int factor);
      
      void multiply(float factor);

      void debug();

      // The color of the pixel.
      color c;

};
#endif
