/**
 * A struct to hold various data resulting from an intersection.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _HITDATA_H
#define _HITDATA_H

class Geometry;

struct HitData
{
    bool hit;

    Vector3f point;

    float t;

    Geometry *object;

};
#endif
