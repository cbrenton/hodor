/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _PNGIMAGE_H
#define _PNGIMAGE_H

#include <string>
#include "img/Image.h"
#include <png++/png.hpp>

using namespace std;

class Pixel;

class PngImage : public Image
{
   public:
      PngImage(int w, int h, string name);

      ~PngImage();

      // Writes the image out to a file.
      void write();

      // Writes a single pixel to the file.
      void writePixel(int x, int y, const Pixel & pix);

      // Closes the file.
      void close();

      // Gets the file extension.
      string getExt();

   protected:
      png::image<png::rgb_pixel_16> *png;
};
#endif
