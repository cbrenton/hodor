/**
 * A geometry object representing a box.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Box.h"
#include "Ray.h"
#include "structs/HitData.h"
#include "Globals.h"

Box::Box(vec3_t c1, vec3_t c2)
{
   vec3_t tmpC1(0.0, 0.0, 0.0);
   vec3_t tmpC2(0.0, 0.0, 0.0);
   for (int dim = 0; dim < 3; dim++)
   {
      tmpC1[dim] = min(c1[dim], c2[dim]);
      tmpC2[dim] = max(c1[dim], c2[dim]);
   }
   b_t.c1 = tmpC1;
   b_t.c2 = tmpC2;
   b_t.left.normal = vec3_t(-1, 0, 0);
   b_t.left.offset = b_t.c1.x();
   b_t.right.normal = vec3_t(1, 0, 0);
   b_t.right.offset = b_t.c2.x();
   b_t.bottom.normal = vec3_t(0, -1, 0);
   b_t.bottom.offset = b_t.c1.y();
   b_t.top.normal = vec3_t(0, 1, 0);
   b_t.top.offset = b_t.c2.y();
   b_t.back.normal = vec3_t(0, 0, -1);
   b_t.back.offset = b_t.c1.z();
   b_t.front.normal = vec3_t(0, 0, 1);
   b_t.front.offset = b_t.c2.z();
}

// Gets the bounding box of the current geometry object.
Box Box::bBox()
{
   Box result(b_t.c1, b_t.c2);
   return result;
}

int Box::hit(Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   return 0;
}

vec3_t Box::getNormal(vec3_t & point)
{
   cout << "box normal" << endl;
   if (closeEnough(point.x(), b_t.left.offset))
   {
      return b_t.left.normal;
   }
   if (closeEnough(point.x(), b_t.right.offset))
   {
      return b_t.right.normal;
   }
   if (closeEnough(point.y(), b_t.bottom.offset))
   {
      return b_t.bottom.normal;
   }
   if (closeEnough(point.y(), b_t.top.offset))
   {
      return b_t.top.normal;
   }
   if (closeEnough(point.z(), b_t.back.offset))
   {
      return b_t.back.normal;
   }
   if (closeEnough(point.z(), b_t.front.offset))
   {
      return b_t.front.normal;
   }
   cerr << "Error: point not on box." << endl;
   return vec3_t(0, 0, 0);
}
