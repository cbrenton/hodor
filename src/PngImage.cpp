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

PngImage::PngImage(int w, int h) : Image(w, h)
{
   curX = 0;
   curY = 0;
}

void PngImage::write()
{
   cout << "filename: " << filename << endl;
   png = new pngwriter(width, height, 0, filename.c_str());
   for (int x = 0; x < width; x++)
   {
      curX = x;
      for (int y = 0; y < height; y++)
      {
         curY = y;
         writePixel(pixelData[x][y]);
      }
   }
   png->close();
}

void PngImage::writePixel(const Pixel & pix)
{
   png->plot(curX, curY, (double)curX / (double)width, (double)curY / (double)height, randFloat());
}

string PngImage::getExt()
{
   return "png";
}
