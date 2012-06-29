/**
 * This stores and writes the pixel data to a .png file.
 * @author Chris Brenton
 * @date 06/22/2011
 */

#ifndef _SFMLIMAGE_H
#define _SFMLIMAGE_H

#include <string>
#include "img/Image.h"
#include <png++/png.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

using namespace std;

class Pixel;

class SFMLImage : public Image
{
   public:
      SFMLImage(int w, int h, string name);

      ~SFMLImage();

      // Writes the image out to the screen.
      void write();

      // Writes a single pixel to the file.
      void writePixel(int x, int y, const Pixel & pix);

      // Closes the window.
      void close();

      // Gets the file extension.
      string getExt();

      sf::Image image;

      sf::RenderWindow *app;

      sf::Sprite *imgSprite;

   protected:
      png::image<png::rgb_pixel_16> *png;
};
#endif
