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
    double ambient;

    double specular;

    double diffuse;

    double reflection;

    double refraction;

    //The index of refraction of the finish.
    double ior;

};
#endif
