/**
 * Holds color data for a single pixel.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "pixel.h"
#include <algorithm>

using namespace std;

Pixel::Pixel(int r, int g, int b)
{
   Pixel((vec_t)r, (vec_t)g, (vec_t)b);
}

Pixel::Pixel(vec_t r, vec_t g, vec_t b)
{
   c.r = r;
   c.g = g;
   c.b = b;
}

void Pixel::clamp()
{
   c.r = min(max(c.r, 0.f), 255.f);
   c.g = min(max(c.g, 0.f), 255.f);
   c.b = min(max(c.b, 0.f), 255.f);
}

void Pixel::add(Pixel other)
{
   c.r += other.c.r;
   c.g += other.c.g;
   c.b += other.c.b;
}

void Pixel::multiply(int factor)
{
   multiply((vec_t)factor);
}

void Pixel::multiply(vec_t factor)
{
   c.r *= factor;
   c.g *= factor;
   c.b *= factor;
}
