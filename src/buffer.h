/**
 * Buffer and buffer3 structs.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _BUFFER_H
#define _BUFFER_H

#include "vector.h"
#include "img/Image.h"
#include "img/PngImage.h"

struct buffer
{
   vec_t *data;
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
inline vec_t * lookup(buffer *buf, int x, int y);
inline vec3 * lookup(buffer3 *buf, int x, int y);
inline void printToFile(std::string filename);

inline void initBuffer(buffer *buf, int w, int h)
{
   buf->data = new vec_t[w * h];
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
         vec_t val = 1.f;
         if (y % 10 < 5)
            val = 0.f;
         pix->v[0] = val;
         pix->v[1] = 0.f;
         pix->v[2] = 0.f;
      }
   }
}

inline vec_t * lookup(buffer *buf, int x, int y)
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
   Image *image = new PngImage(buf->w, buf->h, filename.c_str());
   for (int x = 0; x < buf->w; x++)
   {
      for (int y = 0; y < buf->h; y++)
      {
         image->writePixel(x, y, lookup(buf, x, y));
      }
   }
   image->close();
   delete image;
}

#endif
