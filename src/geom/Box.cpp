/**
 * A geometry object representing a box.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Box.h"
#include "Ray.h"
#include "structs/HitData.h"

Box::Box(Vector3f c1, Vector3f c2)
{
   b_t.c1 = c1;
   b_t.c2 = c2;
   b_t.left.normal = Vector3f(1, 0, 0);
   b_t.left.offset = c1.x();
   b_t.right.normal = Vector3f(-1, 0, 0);
   b_t.right.offset = -c2.x();
   b_t.bottom.normal = Vector3f(0, 1, 0);
   b_t.bottom.offset = c1.y();
   b_t.top.normal = Vector3f(0, -1, 0);
   b_t.top.offset = -c2.y();
   b_t.front.normal = Vector3f(0, 0, 1);
   b_t.front.offset = c1.z();
   b_t.back.normal = Vector3f(0, 0, -1);
   b_t.back.offset = -c2.z();
}

// Gets the bounding box of the current geometry object.
Box Box::bBox()
{
   return Box(b_t.c1, b_t.c2);
}

int Box::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   float tNear = minT - 1;
   float tFar = maxT + 1;

   // For the pair of planes in each dimension.
   for (int i = 0; i < 3; i++)
   {
      // Component of the ray's direction in the current dimension.
      float xD = ray.dir[i];
      // Component of the ray's origin in the current dimension.
      float xO = ray.point[i];
      // Component of the mininum plane location in the current dimension.
      float xL = min(b_t.c1[i], b_t.c2[i]);
      // Component of the maxinum plane location in the current dimension.
      float xH = max(b_t.c1[i], b_t.c2[i]);
      // If direction in current dimension is 0, ray is parallel to planes.
      if (xD == 0)
      {
         // If ray origin is not between the planes.
         if (xO < xL || xO > xH)
         {
            return 0;
         }
      }
      // Else the ray is not parallel to the planes.
      else
      {
         // Calculate tMin and tMax.
         float t1 = (xL - xO) / xD;
         float t2 = (xH - xO) / xD;
         if (t1 > t2)
         {
            swap(t1, t2);
         }
         if (t1 > tNear)
         {
            tNear = t1;
         }
         if (t2 < tFar)
         {
            tFar = t2;
         }
         if (tNear > tFar)
         {
            return 0;
         }
         if (tFar < 0)
         {
            return 0;
         }
      }
   }
   *t = tNear;

   data->hit = 1;
   data->point = ray.point + ray.dir * (*t);
   data->t = (*t);
   data->object = this;
   return 1;

}

Vector3f Box::getNormal(const Vector3f & point)
{
   return Vector3f();
}
