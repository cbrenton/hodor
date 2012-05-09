/**
 * Hit data struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _HIT_DATA_H
#define _HIT_DATA_H

struct hitData
{
   vec3 albedo;
   vec3 kA, kD, kS;
   vec_t rough;
   vec_t illum;
};

#endif
