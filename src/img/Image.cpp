/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "img/Image.h"
#include <stdio.h>

Image::Image(int w, int h) :
   width(w), height(h)
{
   // Initialize pixelData.
   pixelData = new Pixel*[width];
   for (int x = 0; x < width; x++)
   {
      pixelData[x] = new Pixel[height];
      for (int y = 0; y < height; y++)
      {
         pixelData[x][y] = Pixel(0.f, 0.f, 0.f);
      }
   }
}

Image::~Image()
{
   // Delete pixelData.
   for (int x = 0; x < width; x++)
   {
      delete[] pixelData[width];
   }
}

string Image::getExt()
{
   return "out";
}

unsigned char * Image::getPixelBuffer()
{
   unsigned char *ret = new unsigned char[width * height * 3];
   for (int i = 0; i < width * height * 3; i+=3)
   {
      int y = (i / 3) / width;
      int x = (i / 3) % height;
      ret[i] = (unsigned char)(min(pixelData[x][y].c.r * 256.0, 255.0));
      ret[i + 1] = (unsigned char)(min(pixelData[x][y].c.g * 256.0, 255.0));
      ret[i + 2] = (unsigned char)(min(pixelData[x][y].c.b * 256.0, 255.0));
   }
   return ret;
}
