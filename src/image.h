/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include <string>
#include <png++/png.hpp>
#include "glm/glm.hpp"

using namespace std;

class Image
{
   public:
      Image(int w, int h, string fn);

      ~Image();

      // The width in pixels of the image.
      int width;

      // The height in pixels of the image.
      int height;

      // The name of the file to be output (minus the file extension).
      string filename;

      // Writes the image out to a file.
      void write();

      // The pixel data currently stored in the image.
      png::image<png::rgb_pixel> *img;

      // Writes a single pixel to the file.
      void setPixel(int x, int y, glm::vec3 *pix);
};
#endif
