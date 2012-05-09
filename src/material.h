/**
 * Material struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _MATERIAL_H
#define _MATERIAL_H

#include <string>
#include "buffer.h"
#include "vector.h"

struct material
{
   buffer3 *albedo;
   std::string name;
   vec3 kA, kD, kS;
   vec_t rough;
   vec_t illum;
};

extern material * matFromFile(std::string filename);
extern void debug(material *mat);

#endif
