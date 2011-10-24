/**
 * The pigment of the object.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _PIGMENT_T_H
#define _PIGMENT_T_H

#include "structs/color_t.h"
#include "img/PngImage.h"

struct pigment_t
{
   color_t c;
   
   float f;

   bool hasTex;

   PngImage * tex;

};
#endif
