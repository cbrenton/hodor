/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "img/Image.h"
#include "Pixel.h"

Image::Image(int w, int h) :
   width(w), height(h)
{
   // Initialize pixelData.
   pixelData = new Pixel*[width];
   for (int x = 0; x < width; x++) {
      pixelData[x] = new Pixel[height];
      for (int y = 0; y < height; y++) {
         pixelData[x][y] = Pixel(0.0, 0.0, 0.0);
      }
   }
}

Image::~Image()
{
   // Delete pixelData.
   for (int x = 0; x < width; x++) {
      delete[] pixelData[width];
   }
}

string Image::getExt()
{
   return "out";
}
