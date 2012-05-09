/**
 * This stores and writes the pixel data to an image.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include <string>
using namespace std;

class Pixel;

class Image
{
   public:
      Image(int w, int h);

      virtual ~Image();

      // The width in pixels of the image.
      int width;

      // The height in pixels of the image.
      int height;

      // The name of the file to be output (minus the file extension).
      string filename;

      // Writes the image out to a file.
      virtual void write() {};

      // The pixel data currently stored in the image.
      Pixel **pixelData;

      // Writes a single pixel to the file.
      virtual void writePixel(int x, int y, const Pixel & pix) {};

      // Closes the file.
      virtual void close() {};

      // Gets the file extension.
      virtual string getExt();

      // Get the pixel data in an OpenGL-readable buffer.
      virtual unsigned char *getPixelBuffer();

};
#endif
