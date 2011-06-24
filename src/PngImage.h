/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _PNGIMAGE_H
#define _PNGIMAGE_H

#include <string>
#include "Image.h"
#include <pngwriter.h>

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
      pngwriter *png;

};
#endif
