/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include <iostream>
#include "PngImage.h"
#include "Pixel.h"
#include "Globals.h"

using namespace std;

PngImage::PngImage(int w, int h, string name) : Image(w, h)
{
   filename = name;
   curX = 0;
   curY = 0;
   cout << "filename: " << filename << endl;
   png = new pngwriter(width, height, 0, filename.c_str());
}

void PngImage::write()
{
   /*
   for (int x = 0; x < width; x++)
   {
      curX = x;
      for (int y = 0; y < height; y++)
      {
         curY = y;
         writePixel(pixelData[x][y]);
      }
   }
   */
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
