/**
 * This stores and writes the pixel data to a .tga file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _TGAIMAGE_H
#define _TGAIMAGE_H

#include <string>
#include "img/Image.h"
using namespace std;

class Pixel;

class TgaImage : public Image
{
   public:
      TgaImage(int w, int h);

      // Writes the image out to a file.
      void write();

      // Writes a single pixel to the file.
      void writePixel(const Pixel & pix);

      // Gets the file extension.
      string getExt();

};
#endif
