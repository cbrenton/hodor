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

PngImage::PngImage(string name) : Image(1, 1)
{
   filename = name;
   /*
   png = new pngwriter();
   png->readfromfile(filename.c_str());
   width = png->getwidth();
   height = png->getheight();
   cout << "Read image of dimensions (" << width << ", " << height <<
      ") from file " << filename << ".\n";
   png->close();
   */
}

PngImage::~PngImage()
{
   delete png;
}

color_t PngImage::getPixel(int x, int y)
{
   color_t result = {};
   result.r = png->read(x, y, 0);
   result.g = png->read(x, y, 0);
   result.b = png->read(x, y, 0);
   return result;
}

void PngImage::write()
{
}

void PngImage::writePixel(int x, int y, Pixel & pix)
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
