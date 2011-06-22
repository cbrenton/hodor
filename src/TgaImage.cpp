/**
 * This stores and writes the pixel data to a .tga file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include "TgaImage.h"
#include "Pixel.h"
#include <iostream>

using namespace std;

TgaImage::TgaImage(int w, int h) : Image(w, h)
{
}

void TgaImage::write()
{
   cout << "writing out tga with dimensions " << width << "x" << height << endl;
}

void TgaImage::writePixel(const Pixel & pix)
{
}

string TgaImage::getExt()
{
   return "tga";
}
