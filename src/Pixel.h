/**
 * Holds color_t data for a single pixel.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PIXEL_H
#define _PIXEL_H

#include "structs/color_t.h"

class Pixel
{
   public:
      Pixel() {};

      Pixel(double r, double g, double b);

      void clamp();

      void add(Pixel other);

      void multiply(double factor);

      // The color_t of the pixel.
      color_t c;

};
#endif
