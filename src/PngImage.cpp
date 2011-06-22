/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include "PngImage.h"
#include "Pixel.h"
#include <iostream>

using namespace std;

PngImage::PngImage(int w, int h) : Image(w, h)
{
}

void PngImage::write()
{
   pngwriter png(width, height, 0, filename.c_str());
}

void PngImage::writePixel(const Pixel & pix)
{
}

string PngImage::getExt()
{
   return "png";
}
