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
   glm::vec3 pt;
   float alpha, beta, gamma;
   float t;
   int objIndex;
   glm::vec3 albedo;
   glm::vec3 kA, kD, kS;
   float rough;
   float illum;
};

#endif
