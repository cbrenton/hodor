/**
 * Holds color data for a single pixel.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PIXEL_H
#define _PIXEL_H

#include "structs/color.h"

class Pixel
{
   public:
      Pixel() {};

      Pixel(int r, int g, int b);
      
      Pixel(double r, double g, double b);

      void clamp();

      void add(Pixel other);

      void multiply(int factor);
      
      void multiply(double factor);

      // The color of the pixel.
      color c;

};
#endif
