/**
 * Material struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _MATERIAL_H
#define _MATERIAL_H

#include <cstring>
#include <Magick++.h>
#include <string>

#include "buffer.h"
#include "glm/glm.hpp"
#include "globals.h"
#include "objLoader.h"

class material
{
   public:
      material();
      material(obj_material *m);
      ~material();
      glm::vec3 getAmb(int x, int y);
      inline glm::vec3 getDiff() {return kD;};
      inline glm::vec3 getSpec() {return kS;};
   protected:
      glm::vec3 kA, kD, kS;
      float rough;
      float illum;
      Magick::Image tex;
      bool hasTex;
      std::string name;
};

/*
struct material
{
   //buffer3 *tex;
   ILuint tex;
   bool hasTex;
   std::string name;
   glm::vec3 kA, kD, kS;
   float rough;
   float illum;
};

inline material * matFromObj(obj_material *m)
{
   material *result = new material();
   result->name = m->name;
   result->kA = vec3(m->amb[0], m->amb[1], m->amb[2]);
   result->kD = vec3(m->diff[0], m->diff[1], m->diff[2]);
   result->kS = vec3(m->spec[0], m->spec[1], m->spec[2]);
   result->rough = 1.f / m->shiny;
   result->tex = new buffer3();

   initBuffer3(result->tex, m->texture_filename);
   for (int i = 0; i < result->tex->w; i++)
   {
      for (int j = 0; j < result->tex->h; j++)
      {
         vec3 * dataPix = lookup(result->tex, i, j);
         dataPix->x = result->kD[0];
         dataPix->y = result->kD[1];
         dataPix->z = result->kD[2];
      }
   }

   return result;
}

extern material * matFromFile(std::string filename);
extern void debug(material *mat);
*/

#endif
