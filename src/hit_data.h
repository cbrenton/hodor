/**
 * Hit data struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _HIT_DATA_H
#define _HIT_DATA_H

struct hit_data
{
   // TODO: Remove most of these fields.
   int hit;
   vec3 pt;
   float t;
   int objIndex;
   vec3 albedo;
   vec3 kA, kD, kS;
   vec_t rough;
   vec_t illum;
};

#endif
