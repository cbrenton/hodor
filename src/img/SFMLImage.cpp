/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#include <iostream>
#include "img/SFMLImage.h"
#include "Pixel.h"
#include "Globals.h"

using namespace std;

SFMLImage::SFMLImage(int w, int h, string name) : Image(w, h)
{
   filename = name;
   app = new sf::RenderWindow(sf::VideoMode(800, 600, 32), "SFML Graphics");
   cout << "filename: " << filename << endl;
   png = new png::image<png::rgb_pixel_16>(width, height);
   image = sf::Image();
   imgSprite = new sf::Sprite(image);
}

SFMLImage::~SFMLImage()
{
   delete png;
}

void SFMLImage::write()
{
   png->write(filename);
}

void SFMLImage::writePixel(int x, int y, const Pixel & pix)
{
   // Convert colors from double to uint8_t without overflow.
   COLOR_T r = (COLOR_T)(min(pix.c.r * COLOR_RANGE, COLOR_RANGE));
   COLOR_T g = (COLOR_T)(min(pix.c.g * COLOR_RANGE, COLOR_RANGE));
   COLOR_T b = (COLOR_T)(min(pix.c.b * COLOR_RANGE, COLOR_RANGE));

   // Adjust the x and y coordinates for libpng++.
   int correctY = png->get_height() - 1 - y;
   int correctX = x;

   (*png)[correctY][correctX] = png::rgb_pixel_16(r, g, b);

   if (y == 0)
   {
      write();
      if (!image.LoadFromFile(filename))
      {
         // Error...
      }
      imgSprite->SetImage(image);
      app->Draw(*imgSprite);
   }
}

void SFMLImage::close()
{
   write();
}

string SFMLImage::getExt()
{
   return "png";
}
