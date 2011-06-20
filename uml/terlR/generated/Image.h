#ifndef _IMAGE_H
#define _IMAGE_H


#include <string>
using namespace std;

class Pixel;

//This stores and writes the pixel data to an image.
class Image
{
  public:
    //The height in pixels of the image.
    int height;

    //The width in pixels of the image.
    int width;

    //The name of the file to be output (minus the file extension).
    string filename;

    //Writes the image out to a file.
    void write();

    //The pixel data currently stored in the image.
    Pixel **pixelData;

};
#endif
