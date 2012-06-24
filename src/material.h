/**
 * Material struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _MATERIAL_H
#define _MATERIAL_H

#include <string>
#include "buffer.h"
#include "glm/glm.hpp"
#include "globals.h"

struct material
{
   buffer3 *albedo;
   std::string name;
   glm::vec3 kA, kD, kS;
   float rough;
   float illum;
};

extern material * matFromFile(std::string filename);
extern void debug(material *mat);

#endif
