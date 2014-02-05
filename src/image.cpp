/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "image.h"
#include <stdio.h>

Image::Image(int w, int h, string fn) :
   width(w), height(h), filename(fn)
{
   img = new png::image<png::rgb_pixel>(w, h);
}

Image::~Image()
{
   delete img;
}

void Image::write()
{
   img->write(filename);
}

void Image::setPixel(int x, int y, glm::vec3 *pix)
{
   img->set_pixel(x, y, png::rgb_pixel(pix->x, pix->y, pix->z));
}
