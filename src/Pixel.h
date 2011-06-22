/**
 * Holds color data for a single pixel.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PIXEL_H
#define _PIXEL_H

#include "color.h"

class Pixel
{
   public:
      // The color of the pixel.
      color clr;

   private:
      // The alpha value of the current pixel. Will never be used in most image types.
      double alpha;

};
#endif
