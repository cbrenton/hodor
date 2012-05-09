/**
 * Holds color data for a single pixel.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PIXEL_H
#define _PIXEL_H

#include "vector.h"
#include "structs/color.h"

class Pixel
{
   public:
      Pixel() {};

      Pixel(int r, int g, int b);
      
      Pixel(vec_t r, vec_t g, vec_t b);

      void clamp();

      void add(Pixel other);

      void multiply(int factor);
      
      void multiply(vec_t factor);

      // The color of the pixel.
      color c;

};
#endif
