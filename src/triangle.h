/**
 * Triangle struct.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include "vertex.h"
#include "material.h"
#include "objLoader.h"

struct triangle
{
   vertex *pts[3];
   material *mat;
   int matNdx;
};

inline void debug(triangle *t)
{
   printf("triangle:\n");
   printf("\t");
   mPRLN_VEC(t->pts[0]->coord);
   printf("\t");
   mPRLN_VEC(t->pts[1]->coord);
   printf("\t");
   mPRLN_VEC(t->pts[2]->coord);
}

inline void initTri(triangle *t)
{
   t->pts[0] = new vertex;
   t->pts[1] = new vertex;
   t->pts[2] = new vertex;
}

inline triangle * faceToTri(obj_face *f, objLoader *l)
{
   if (f->vertex_count != 3)
   {
      fprintf(stderr, "Invalid face: %d vertices.\n", f->vertex_count);
      exit(EXIT_FAILURE);
   }
   triangle *ret = new triangle;
   initTri(ret);
   // Assign face properties to triangle pointer.
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         ret->pts[i]->coord[j] = (float)l->vertexList[f->vertex_index[i]]->e[j];
         /*
         if (j != 2)
         {
            ret->pts[i]->texCoord[j] = (float)l->textureList[f->texture_index[i]]->e[j];
         }
         ret->pts[i]->normal[j] = (float)l->normalList[f->normal_index[i]]->e[j];
         */
      }
   }
   ret->matNdx = f->material_index;
   return ret;
}

inline triangle * quadToTri(obj_face *f, objLoader *l)
{
   if (f->vertex_count != 4)
   {
      fprintf(stderr, "Invalid face: %d vertices.\n", f->vertex_count);
      exit(EXIT_FAILURE);
   }
   triangle *ret = new triangle[2];
   initTri(ret);
   // Assign face properties to triangle pointer.
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         ret[0].pts[i]->coord[j] = (float)l->vertexList[f->vertex_index[i]]->e[j];
         /*
         if (j != 2)
         {
            //ret[0].pts[i]->texCoord[j] = (float)l->textureList[f->texture_index[i]]->e[j];
         }
         //ret[0].pts[i]->normal[j] = (float)l->normalList[f->normal_index[i]]->e[j];
         */
      }
   }
   // Add the point at index 0 to the second triangle.
   for (int j = 0; j < 3; j++)
   {
      ret[0].pts[0]->coord[j] = (float)l->vertexList[f->vertex_index[0]]->e[j];
      if (j != 2)
      {
         ret[0].pts[0]->texCoord[j] = (float)l->textureList[f->texture_index[0]]->e[j];
      }
      ret[1].pts[0]->normal[j] = (float)l->normalList[f->normal_index[0]]->e[j];
   }
   /*
   // Add the other two points.
   for (int i = 2; i < 4; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         ret[1].pts[i]->coord[j] = (float)l->vertexList[f->vertex_index[i]]->e[j];
         if (j != 2)
         {
            ret[1].pts[i]->texCoord[j] = (float)l->textureList[f->texture_index[i]]->e[j];
         }
         ret[1].pts[i]->normal[j] = (float)l->normalList[f->normal_index[i]]->e[j];
      }
   }
   */

   ret[0].matNdx = ret[1].matNdx = f->material_index;
   return ret;
}

#endif
