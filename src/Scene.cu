/**
 * This holds scene geometry data. ray_t casting and shading take place here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include <iostream>
#include "Scene.h"
#include "parse/nyuparser.h"
#include "Globals.h"

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
   NYUParser *parser = new NYUParser;
   parser->parse(input, *curScene);
   /*
      for (int geomNdx = 0; geomNdx < (int)curScene->geometry.size(); geomNdx++)
      {
      curScene->geometry[geomNdx]->debug();
      }
    */
   delete parser;
   return curScene;
}

/**
 * Checks if a ray intersects any geometry in the scene, using structs.
 * @returns true if an intersection is found.
 */
bool Scene::gpuHit(ray_t & ray, hit_t *data)
{
   // INITIALIZE closestT to MAX_DIST + 0.1
   float closestT = MAX_DIST + 0.1f;
   // INITIALIZE closestData to empty hit_t
   hit_t *closestData = new hit_t();

   // Find hit for boxes.
   // FOR each item in boxes
   for (int boxNdx = 0; boxNdx < (int)boxes.size(); boxNdx++)
   {
      float boxT = -1;
      hit_t *boxData = new hit_t();
      // IF current item is hit by ray
      if (box_hit(boxes[boxNdx], ray, &boxT, boxData) != 0)
      {
         // IF intersection is closer than closestT
         if (boxT < closestT)
         {
            // SET closestT to intersection
            closestT = boxT;
            // SET closestData to intersection data
            *closestData = *boxData;
            closestData->objIndex = boxNdx;
         }
         // ENDIF
      }
      // ENDIF
      delete boxData;
   }

   // Find hit for planes.
   // FOR each item in geometry
   for (int planeNdx = 0; planeNdx < (int)planes.size(); planeNdx++)
   {
      float planeT = -1;
      hit_t *planeData = new hit_t();
      // IF current item is hit by ray
      if (plane_hit(planes[planeNdx], ray, &planeT, planeData) != 0)
      {
         // IF intersection is closer than closestT
         if (planeT < closestT)
         {
            // SET closestT to intersection
            closestT = planeT;
            // SET closestData to intersection data
            *closestData = *planeData;
            closestData->objIndex = planeNdx;
         }
         // ENDIF
      }
      // ENDIF
      delete planeData;
   }

   // Find hit for spheres.
   // FOR each item in spheres
   for (int sphereNdx = 0; sphereNdx < (int)spheres.size(); sphereNdx++)
   {
      float sphereT = -1;
      hit_t *sphereData = new hit_t();
      // IF current item is hit by ray
      if (sphere_hit(spheres[sphereNdx], ray, &sphereT, sphereData) != 0)
      {
         // IF intersection is closer than closestT
         if (sphereT < closestT)
         {
            // SET closestT to intersection
            closestT = sphereT;
            // SET closestData to intersection data
            *closestData = *sphereData;
            closestData->objIndex = sphereNdx;
         }
         // ENDIF
      }
      // ENDIF
      delete sphereData;
   }

   // Find hit for triangles.
   // FOR each item in triangles
   for (int triNdx = 0; triNdx < (int)triangles.size(); triNdx++)
   {
      float triT = -1;
      hit_t *triData = new hit_t();
      // IF current item is hit by ray
      if (triangle_hit(triangles[triNdx], ray, &triT, triData) != 0)
      {
         // IF intersection is closer than closestT
         if (triT < closestT)
         {
            // SET closestT to intersection
            closestT = triT;
            // SET closestData to intersection data
            *closestData = *triData;
            closestData->objIndex = triNdx;
         }
         // ENDIF
      }
      // ENDIF
      delete triData;
   }

   // ENDFOR
   // IF data is not null
   if (data != NULL)
   {
      // SET data to closestData
      *data = *closestData;
   }
   // ENDIF
   delete closestData;
   // RETURN true if closestT is less than or equal to MAX_DIST
   return (closestT <= MAX_DIST);
}

/**
 * Checks if a ray intersects any geometry in the scene, using Geometry.
 * @returns true if an intersection is found.
 */
bool Scene::hit(ray_t & ray, hit_t *data)
{
   // INITIALIZE closestT to MAX_DIST + 0.1
   float closestT = MAX_DIST + 0.1f;
   // INITIALIZE closestData to empty hit_t
   hit_t *closestData = new hit_t();
   // FOR each item in geometry
   for (int geomNdx = 0; geomNdx < (int)geometry.size(); geomNdx++)
   {
      float geomT = -1;
      hit_t *geomData = new hit_t();
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
      delete geomData;
   }
   // ENDFOR
   // IF data is not null
   if (data != NULL)
   {
      // SET data to closestData
      *data = *closestData;
   }
   // ENDIF
   delete closestData;
   // RETURN true if closestT is less than or equal to MAX_DIST
   return (closestT <= MAX_DIST);
}

Pixel** Scene::castRays(ray_t **rays, int width, int height, int depth)
{
   sphere_t **sphereArray = thrust::raw_pointer_cast(&spheres[0]);
   hit_t *results = new hit_t[width * height];
   //hitSpheres(*sphereArray, spheres.size(), rays, results);



   cout << "cuda_test: (" << width << ", " << height << ")" << endl;
   ray_t **r_d;
   size_t size = width * height * sizeof(ray_t);
   cudaMalloc((void **) &r_d, size);
   cudaMemcpy(r_d, rays, size, cudaMemcpyHostToDevice);
   int block_size = 4;
   int n_blocks = width * height / block_size +
      (width * height % block_size > 0 ? 1 : 0);
   cout << "n_blocks: " << n_blocks << endl;

   initPrintf();

   cuda_test <<< block_size, n_blocks >>> (r_d, width, height);

   cudaMemcpy(rays, r_d, sizeof(ray_t)*width*height, cudaMemcpyDeviceToHost);

   endPrintf();

   cudaFree(r_d);


   return NULL;
}

// Casts a ray into the scene and returns a correctly color_ted pixel.
Pixel Scene::castRay(ray_t & ray, int depth)
{
   Pixel result(0.0, 0.0, 0.0);
   Pixel reflectPix(0.0, 0.0, 0.0);
   Pixel refractPix(0.0, 0.0, 0.0);
   hit_t rayData;
   //if (hit(ray, &rayData))
   if (gpuHit(ray, &rayData))
   {
      result = shade(&rayData, ray.dir);
      if (useGPU)
      {
         if (rayData.reflect != NULL && depth > 0)
         {
            //pigment_t hitP = {};
            finish_t hitF = {};
            vec3_t hitNormal(0.0, 0.0, 0.0);
            box_t *b_t;
            plane_t *p_t;
            sphere_t *s_t;
            triangle_t *t_t;
            switch (rayData.hitType) {
            case BOX_HIT:
               b_t = boxes[rayData.objIndex];
               //hitP = b_t->p;
               hitF = b_t->f;
               hitNormal = box_normal(b_t, &rayData);
               break;
            case PLANE_HIT:
               p_t = planes[rayData.objIndex];
               //hitP = p_t->p;
               hitF = p_t->f;
               hitNormal = plane_normal(p_t);
               break;
            case SPHERE_HIT:
               s_t = spheres[rayData.objIndex];
               //hitP = s_t->p;
               hitF = s_t->f;
               hitNormal = sphere_normal(s_t, &rayData);
               break;
            case TRIANGLE_HIT:
               t_t = triangles[rayData.objIndex];
               //hitP = t_t->p;
               hitF = t_t->f;
               hitNormal = triangle_normal(t_t);
               break;
            }

            vec3_t reflectVec = *rayData.reflect;
            reflectVec.normalize();
            vec3_t reflectOrig = reflectVec * EPSILON;
            reflectOrig += rayData.point;
            ray_t reflectray_t = {reflectOrig, reflectVec};

            reflectPix = castRay(reflectray_t, depth - 1);

            reflectPix.multiply(hitF.reflection);
            result.multiply(1 - hitF.reflection);
            result.add(reflectPix);
         }
      }

   }
   return result;
}

// Calculates proper shading at the current point.
Pixel Scene::shade(hit_t *data, vec3_t view)
{
   Pixel result(0.0, 0.0, 0.0);
   pigment_t hitP = {};
   finish_t hitF = {0};
   vec3_t hitNormal(0.0, 0.0, 0.0);
   box_t *b_t;
   plane_t *p_t;
   sphere_t *s_t;
   triangle_t *t_t;
   switch (data->hitType) {
   case BOX_HIT:
      b_t = boxes[data->objIndex];
      hitP = b_t->p;
      hitF = b_t->f;
      hitNormal = box_normal(b_t, data);
      break;
   case PLANE_HIT:
      p_t = planes[data->objIndex];
      hitP = p_t->p;
      hitF = p_t->f;
      hitNormal = plane_normal(p_t);
      break;
   case SPHERE_HIT:
      s_t = spheres[data->objIndex];
      hitP = s_t->p;
      hitF = s_t->f;
      hitNormal = sphere_normal(s_t, data);
      break;
   case TRIANGLE_HIT:
      t_t = triangles[data->objIndex];
      hitP = t_t->p;
      hitF = t_t->f;
      hitNormal = triangle_normal(t_t);
      break;
   }

   for (int lightNdx = 0; lightNdx < (int)lights.size(); lightNdx++)
   {
      Light *curLight = lights[lightNdx];
      // Ambient.
      result.c.r += (hitF.ambient*hitP.c.r) * curLight->r;
      result.c.g += (hitF.ambient*hitP.c.g) * curLight->g;
      result.c.b += (hitF.ambient*hitP.c.b) * curLight->b;

      // Cast light feeler ray.
      ray_t feeler;
      feeler.dir = curLight->location - data->point;
      feeler.dir.normalize();
      feeler.point = feeler.dir * EPSILON;
      feeler.point += data->point;

      hit_t tmpHit;

      // If feeler hits any object, current point is in shadow.
      bool isShadow = gpuHit(feeler, &tmpHit);

      if (!isShadow)
      {
         // Diffuse.
         vec3_t n = hitNormal;
         n.normalize();
         vec3_t l = curLight->location - data->point;
         l.normalize();
         float nDotL = n.dot(l);
         nDotL = min(nDotL, 1.0f);

         if (nDotL > 0)
         {
            result.c.r += hitF.diffuse*hitP.c.r * nDotL * curLight->r;
            result.c.g += hitF.diffuse*hitP.c.g * nDotL * curLight->g;
            result.c.b += hitF.diffuse*hitP.c.b * nDotL * curLight->b;
         }

         // Specular (Phong).
         vec3_t r = mReflect(l, n);
         r.normalize();
         vec3_t v = view;
         v.normalize();
         float rDotV = r.dot(v);
         rDotV = (float)pow(rDotV, 1.0f / hitF.roughness);
         rDotV = min(rDotV, 1.0f);

         if (rDotV > 0)
         {
            result.c.r += hitF.specular*hitP.c.r * rDotV * curLight->r;
            result.c.g += hitF.specular*hitP.c.g * rDotV * curLight->g;
            result.c.b += hitF.specular*hitP.c.b * rDotV * curLight->b;
         }
      }
   }
   return result;
}

/*
   vec3_t Scene::reflect(vec3_t d, vec3_t n)
   {
   return n * (2 * (-d.dot(n))) + d;
   }
 */
