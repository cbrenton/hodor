/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include <iostream>
#include "img/PngImage.h"
#include "Pixel.h"
#include "Globals.h"

using namespace std;

PngImage::PngImage(int w, int h, string name) : Image(w, h)
{
   filename = name;
   cout << "filename: " << filename << endl;
   png = new pngwriter(width, height, 0, filename.c_str());
}

PngImage::~PngImage()
{
   delete png;
}

void PngImage::write()
{
}

void PngImage::writePixel(int x, int y, const Pixel & pix)
{
   png->plot(x, y, pix.c.r, pix.c.g, pix.c.b);
}

void PngImage::close()
{
   png->close();
}

string PngImage::getExt()
{
   return "png";
}
