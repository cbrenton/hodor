/**
 * Buffer and buffer3 structs.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _BUFFER_H
#define _BUFFER_H

#include "glm/glm.hpp"
#include "img/image.h"

using namespace glm;

struct buffer
{
   float *data;
   int w, h;
};

struct buffer3
{
   vec3 *data;
   int w, h;
};

inline void initBuffer(buffer *buf, int w, int h);
inline void initBuffer3(buffer3 *buf, int w, int h);
inline void drawStripes(buffer3 *buf, int w, int h);
inline float * lookup(buffer *buf, int x, int y);
inline vec3 * lookup(buffer3 *buf, int x, int y);
inline void printToFile(std::string filename);

inline void initBuffer(buffer *buf, int w, int h)
{
   buf->data = new float[w * h];
   buf->w = w;
   buf->h = h;
}

inline void initBuffer3(buffer3 *buf, int w, int h)
{
   buf->data = new vec3[w * h];
   buf->w = w;
   buf->h = h;
}

// TODO: Make this not inlined (too many loops).
inline void drawStripes(buffer3 *buf, int w, int h)
{
   initBuffer3(buf, w, h);
   for (int x = 0; x < w; x++)
   {
      for (int y = 0; y < h; y++)
      {
         vec3 *pix = lookup(buf, x, y);
         float val = 1.f;
         if (y % 10 < 5)
            val = 0.f;
         pix->x = val;
         pix->y = 0.f;
         pix->z = 0.f;
      }
   }
}

inline float * lookup(buffer *buf, int x, int y)
{
   int newX = x;
   int newY = y;
   if (x < 0)
      newX = 0;
   if (x > buf->w)
      newX = buf->w;
   if (y < 0)
      newY = 0;
   if (y > buf->h)
      newY = buf->h;
   return &buf->data[newX * buf->h + newY];
}

inline vec3 * lookup(buffer3 *buf, int x, int y)
{
   int newX = x;
   int newY = y;
   if (x < 0)
      newX = 0;
   if (x > buf->w)
      newX = buf->w;
   if (y < 0)
      newY = 0;
   if (y > buf->h)
      newY = buf->h;
   return &buf->data[newX * buf->h + newY];
}

inline void printToFile(buffer3 *buf, std::string filename)
{
   Image *image = new Image(buf->w, buf->h, filename.c_str());
   for (int x = 0; x < buf->w; x++)
   {
      for (int y = 0; y < buf->h; y++)
      {
         image->setPixel(x, y, lookup(buf, x, y));
      }
   }
   image->write();
   delete image;
}

#endif
