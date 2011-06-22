/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Image.h"
#include "Pixel.h"

Image::Image(int w, int h) :
   width(w), height(h)
{
   // Initialize pixelData.
}

Image::~Image()
{
   // Delete pixelData.
}

string Image::getExt()
{
   return "out";
}
