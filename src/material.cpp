/**
 * Material struct operations.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#include "material.h"

using namespace glm;

material::material()
{
   hasTex = false;
}

material::material(obj_material *m)
{
   material();
   name = m->name;
   kA = vec3(m->amb[0], m->amb[1], m->amb[2]);
   kD = vec3(m->diff[0], m->diff[1], m->diff[2]);
   kS = vec3(m->spec[0], m->spec[1], m->spec[2]);
   rough = 1.f / m->shiny;
   if (strlen(m->texture_filename) > 0)
   {
      hasTex = true;
      std::string filename = m->texture_filename;
      filename.erase(filename.find_last_not_of(" \n\r\t")+1);
      printf("loading image: %s\n", filename.c_str());
      try
      {
         tex = Magick::Image(filename.c_str());
      }
      catch ( Magick::Exception & error)
      {
         printf("Caught Magick++ exception: %s\n", error.what());
      }
   }
}

material::~material()
{
}

vec3 material::getAmb(int x, int y)
{
   if (!hasTex)
   {
      return kA;
   }
   //return vec3(1.0, 0.0, 0.0);
   Magick::ColorRGB result(tex.pixelColor(x, y));
   vec3 vresult = vec3((float)result.red(), (float)result.green(), (float)result.blue());
   mPRLN_VEC(vresult);
   return vresult;
}
