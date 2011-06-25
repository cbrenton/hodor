/**
 * A geometry object representing a triangle.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "geom/Triangle.h"
#include "Ray.h"
#include "structs/HitData.h"
#include "Globals.h"

// Gets the bounding box of the current geometry object.
Box Triangle::bBox()
{
   float minX, minY, minZ, maxX, maxY, maxZ;
   minX = min3(t_t.c1.x(), t_t.c2.x(), t_t.c3.x());
   maxX = max3(t_t.c1.x(), t_t.c2.x(), t_t.c3.x());
   minY = min3(t_t.c1.y(), t_t.c2.y(), t_t.c3.y());
   maxY = max3(t_t.c1.y(), t_t.c2.y(), t_t.c3.y());
   minZ = min3(t_t.c1.z(), t_t.c2.z(), t_t.c3.z());
   maxZ = max3(t_t.c1.z(), t_t.c2.z(), t_t.c3.z());
   Vector3f c1 = Vector3f(minX, minY, minZ);
   Vector3f c2 = Vector3f(maxX, maxY, maxZ);
   return Box(c1, c2);
}

int Triangle::hit(const Ray & ray, float *t, HitData *data, float minT, float maxT)
{
   float result = -1;
   float bBeta, bGamma, bT;

   Matrix3f A;
   A << t_t.c1.x()-t_t.c2.x(), t_t.c1.x()-t_t.c3.x(), ray.dir.x(),
     t_t.c1.y()-t_t.c2.y(), t_t.c1.y()-t_t.c3.y(), ray.dir.y(),
     t_t.c1.z()-t_t.c2.z(), t_t.c1.z()-t_t.c3.z(), ray.dir.z();
   float detA = A.determinant();

   Matrix3f baryT;
   baryT <<
      t_t.c1.x()-t_t.c2.x(), t_t.c1.x()-t_t.c3.x(), t_t.c1.x()-ray.point.x(),
      t_t.c1.y()-t_t.c2.y(), t_t.c1.y()-t_t.c3.y(), t_t.c1.y()-ray.point.y(),
      t_t.c1.z()-t_t.c2.z(), t_t.c1.z()-t_t.c3.z(), t_t.c1.z()-ray.point.z();

   bT = baryT.determinant() / detA;

   if (bT < 0)
   {
      result = 0;
   }
   else
   {
      Matrix3f baryGamma;
      baryGamma <<
         t_t.c1.x()-t_t.c2.x(), t_t.c1.x()-ray.point.x(), ray.dir.x(),
         t_t.c1.y()-t_t.c2.y(), t_t.c1.y()-ray.point.y(), ray.dir.y(),
         t_t.c1.z()-t_t.c2.z(), t_t.c1.z()-ray.point.z(), ray.dir.z();

      bGamma = baryGamma.determinant() / detA;

      if (bGamma < 0 || bGamma > 1)
      {
         result = 0;
      }
      else
      {
         Matrix3f baryBeta;
         baryBeta <<
            t_t.c1.x()-ray.point.x(), t_t.c1.x()-t_t.c3.x(), ray.dir.x(),
            t_t.c1.y()-ray.point.y(), t_t.c1.y()-t_t.c3.y(), ray.dir.y(),
            t_t.c1.z()-ray.point.z(), t_t.c1.z()-t_t.c3.z(), ray.dir.z();

         bBeta = baryBeta.determinant() / detA;

         if (bBeta < 0 || bBeta > 1 - bGamma)
         {
            result = 0;
         }
      }
   }
  
   if (result != 0)
   {
      result = bT;
   }
   *t = result;
   if (result > EPSILON)
   {
      data->hit = 1;
      data->point = ray.point + ray.dir * (*t);
      data->t = (*t);
      data->object = this;
      return 1;
   }
   return 0;
}

Vector3f Triangle::getNormal(const Vector3f & point)
{
   Vector3f s1 = t_t.c2 - t_t.c1;
   Vector3f s2 = t_t.c3 - t_t.c1;
   return s1.cross(s2);
}
