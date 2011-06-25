/**
 * Holds color data for a single pixel.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "Pixel.h"
#include <algorithm>

using namespace std;

Pixel::Pixel(double r, double g, double b)
{
   c.r = r;
   c.g = g;
   c.b = b;
}

void Pixel::clamp()
{
   c.r = min(max(c.r, 0.0), 1.0);
   c.g = min(max(c.g, 0.0), 1.0);
   c.b = min(max(c.b, 0.0), 1.0);
}

void Pixel::add(Pixel other)
{
   c.r += other.c.r;
   c.g += other.c.g;
   c.b += other.c.b;
}

void Pixel::multiply(double factor)
{
   c.r *= factor;
   c.g *= factor;
   c.b *= factor;
}
