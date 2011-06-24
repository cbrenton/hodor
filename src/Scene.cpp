/**
 * This holds scene geometry data. Ray casting and shading take place here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include <iostream>
#include "Scene.h"
#include "nyuparser.h"

using namespace std;

/**
 * Constructs a bounding volume heirarchy for the scene.
 */
void Scene::constructBVH()
{
}

/**
 * Reads in scene data from a file and returns a new Scene containing the newly
 * stored data.
 * @returns a pointer to the newly created Scene.
 */
Scene* Scene::read(std::fstream & input)
{
   Scene* curScene = new Scene();
   NYUParser parser;
   parser.parse(input, *curScene);
   for (int geomNdx = 0; geomNdx < (int)curScene->geometry.size(); geomNdx++)
   {
      curScene->geometry[geomNdx]->debug();
   }
   return curScene;
}

/**
 * Checks if a ray intersects any geometry in the scene.
 * @returns true if an intersection is found.
 */
bool Scene::hit(const Ray & ray, HitData *data)
{
   // INITIALIZE closestT to MAX_DIST + 0.1
   float closestT = MAX_DIST + 0.1f;
   // INITIALIZE closestData to empty HitData
   HitData *closestData = new HitData();
   // FOR each item in geometry
   for (int geomNdx = 0; geomNdx < (int)geometry.size(); geomNdx++)
   {
      float geomT = -1;
      HitData *geomData = new HitData();
      // IF current item is hit by ray
      if (geometry[geomNdx]->hit(ray, &geomT, geomData) != 0)
      {
         // IF intersection is closer than closestT
         if (geomT < closestT)
         {
            // SET closestT to intersection
            closestT = geomT;
            // SET closestData to intersection data
            *closestData = *geomData;
         }
         // ENDIF
      }
      // ENDIF
   }
   // ENDFOR
   // IF data is not null
   if (data != NULL)
   {
      // SET data to closestData
      *data = *closestData;
   }
   // ENDIF
   // RETURN true if closestT is less than or equal to MAX_DIST
   return (closestT <= MAX_DIST);
}

// Casts a ray into the scene and returns a correctly colored pixel.
Pixel Scene::castRay(const Ray & ray, int depth)
{
   Pixel result(0.0, 0.0, 0.0);
   HitData rayData;
   if (hit(ray, &rayData))
   {
      result = shade(&rayData, ray.dir);
   }
   return result;
}

// Calculates proper shading at the current point.
Pixel Scene::shade(HitData *data, Vector3f view)
{
   Pixel result(0.0, 0.0, 0.0);
   for (int lightNdx = 0; lightNdx < (int)lights.size(); lightNdx++)
   {
      Light *curLight = lights[lightNdx];
      // Ambient.
      result.c.r = (data->object->f.ambient*data->object->p.c.r) * curLight->r;
      result.c.g = (data->object->f.ambient*data->object->p.c.g) * curLight->g;
      result.c.b = (data->object->f.ambient*data->object->p.c.b) * curLight->b;

      // Cast light feeler ray.
      Ray feeler;
      feeler.dir = curLight->location - data->point;
      feeler.dir.normalize();
      feeler.point = data->point + feeler.dir * EPSILON;

      HitData tmpHit;

      // If feeler hits any object, current point is in shadow.
      bool isShadow = hit(feeler, &tmpHit);

      if (!isShadow)
      {
         // Diffuse.
         Vector3f n = data->object->getNormal(data->point);
         n.normalize();
         Vector3f l = curLight->location - data->point;
         l.normalize();
         float nDotL = n.dot(l);
         nDotL = min(nDotL, 1.0f);
         if (nDotL < 0)
         {
            nDotL *= -1;
         }
         
         result.c.r += data->object->f.diffuse*data->object->p.c.r * nDotL *
            curLight->r;
         result.c.g += data->object->f.diffuse*data->object->p.c.g * nDotL *
            curLight->g;
         result.c.b += data->object->f.diffuse*data->object->p.c.b * nDotL *
            curLight->b;
      }
   }
   return result;
}

