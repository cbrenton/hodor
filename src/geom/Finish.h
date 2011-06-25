/**
 * The finish of the object.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _FINISH_H
#define _FINISH_H

class Finish
{
   public:
      Finish() :
         ambient(0.0), specular(0.0), diffuse(0.0), roughness(0.0),
         reflection(0.0), refraction(0.0), ior(0.0)
      {};

      double ambient;

      double specular;

      double diffuse;

      double roughness;

      double reflection;

      double refraction;

      // The index of refraction of the finish.
      float ior;

};
#endif
