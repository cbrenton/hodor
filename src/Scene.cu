/**
 * This holds scene geometry data. ray_t casting and shading take place here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include <iostream>
#include "Scene.h"
#include "parse/nyuparser.h"
#include "Globals.h"
#include "hit_kernel.h"
#include "structs/hitd_t.h"
#include <cutil.h>

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
   curScene->spheresArray = new sphere_t[curScene->spheres.size()];
   for (int i = 0; i < (int)curScene->spheres.size(); i++)
   {
      curScene->spheresArray[i] = *curScene->spheres[i];
   }
   return curScene;
}

/**
 * Checks if a ray intersects any geometry in the scene, using structs.
 * @returns true if an intersection is found.
 */
bool Scene::gpuHit(ray_t & ray, hit_t *data)
{
   /*
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
*/
return false;
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

void Scene::cudaSetup(int chunkSize)
{
   cout << "Allocating device arrays...";
   // Create sphere array on device.
   spheres_size = sizeof(sphere_t) * spheres.size();
   CUDA_SAFE_CALL(cudaMalloc((void**) &spheres_d, spheres_size));
   // Copy spheres to device.
   CUDA_SAFE_CALL(cudaMemcpy(spheres_d, spheresArray, spheres_size,
            cudaMemcpyHostToDevice));

   // Create ray array on device.
   rays_size = chunkSize * sizeof(ray_t);
   CUDA_SAFE_CALL(cudaMalloc((void **) &rays_d, rays_size));

   // Create hit data array on device.
   results_size = chunkSize * sizeof(hitd_t);
   CUDA_SAFE_CALL(cudaMalloc((void **) &results_d, results_size));

   // Create hit data array on host.
   results = new hitd_t[chunkSize];
   cout << "done." << endl;
}

void Scene::cudaCleanup()
{
   cout << "Cleaning up device arrays...";
   cudaFree(spheres_d);
   cudaFree(rays_d);
   cudaFree(results_d);

   delete[] results;
   cout << "done." << endl;
}

/**
 * Casts rays into the scene and returns correctly colored pixels.
 */
Pixel* Scene::castRays(ray_t *rays, int num, int depth)
{
   Pixel *pixels = new Pixel[num];

   for (int i = 0; i < num; i++)
   {
      results[i].hit = 0;
   }
   CUDA_SAFE_CALL(cudaMemcpy(results_d, results, results_size,
            cudaMemcpyHostToDevice));

   // Copy rays to device.
   CUDA_SAFE_CALL(cudaMemcpy(rays_d, rays, rays_size, cudaMemcpyHostToDevice));
   // Calculate block size and number of blocks.
   dim3 dimGrid((int)ceil((float)num / (float)THREADS_PER_BLOCK), 1);
   dim3 dimBlock(THREADS_PER_BLOCK, 1);

   set_spheres <<< dimGrid, dimBlock >>> (spheres_d, spheres.size());
   // Test for intersection.
   cuda_hit <<< dimGrid, dimBlock >>>
      (rays_d, num, results_d);
   //cuda_hit <<< dimGrid, dimBlock >>>
      //(rays_d, num, spheres_d, spheres.size(), results_d);
   // Check for error.
   cudaError_t err = cudaGetLastError();
   if( cudaSuccess != err)
   {
      fprintf(stderr, "Cuda error: %s: %s.\n", "kernel",
            cudaGetErrorString( err) );
      exit(EXIT_FAILURE);
   }

   // Copy hit data to host.
   CUDA_SAFE_CALL(cudaMemcpy(results, results_d, results_size,
            cudaMemcpyDeviceToHost));

   // Color result pixels.
   for (int resultNdx = 0; resultNdx < num; resultNdx++)
   {
      hitd_t curResult = results[resultNdx];
      ray_t curRay = rays[resultNdx];
      if (curResult.hit != 0)
      {
         sphere_t hitSphere = spheresArray[curResult.objIndex];
         pixels[resultNdx] = shade(curResult, curRay);
      }
      else
      {
         pixels[resultNdx] = Pixel(0.0, 0.0, 0.0);
      }
   }

   return pixels;
}

// Calculates proper shading at the current point.
Pixel Scene::shade(hitd_t & data, ray_t & view)
{
   Pixel result(0.0, 0.0, 0.0);
   pigment_t hitP = {};
   finish_t hitF = {0};
   vec3_t hitNormal(0.0, 0.0, 0.0);
   vec3_t dataPoint = view.dir * data.t;
   dataPoint += view.point;
   box_t *b_t;
   plane_t *p_t;
   sphere_t s_t;
   triangle_t *t_t;
   switch (data.hitType) {
   case BOX_HIT:
      b_t = boxes[data.objIndex];
      hitP = b_t->p;
      hitF = b_t->f;
      //hitNormal = box_normal(b_t, data);
      break;
   case PLANE_HIT:
      p_t = planes[data.objIndex];
      hitP = p_t->p;
      hitF = p_t->f;
      //hitNormal = plane_normal(p_t);
      break;
   case SPHERE_HIT:
      s_t = spheresArray[data.objIndex];
      hitP = s_t.p;
      hitF = s_t.f;
      hitNormal = sphere_normal(s_t, dataPoint);
      break;
   case TRIANGLE_HIT:
      t_t = triangles[data.objIndex];
      hitP = t_t->p;
      hitF = t_t->f;
      //hitNormal = triangle_normal(t_t);
      break;
   default:
      cerr << "Invalid intersection type." << endl;
   }

   for (int lightNdx = 0; lightNdx < (int)lights.size(); lightNdx++)
   {
      Light *curLight = lights[lightNdx];
      // Ambient.
      result.c.r += (hitF.ambient*hitP.c.r) * curLight->r;
      result.c.g += (hitF.ambient*hitP.c.g) * curLight->g;
      result.c.b += (hitF.ambient*hitP.c.b) * curLight->b;

      // Diffuse.
      vec3_t n = hitNormal;
      n.normalize();
      vec3_t dataPoint = data.point.toHost();
      vec3_t l = curLight->location - dataPoint;
      l.normalize();
      float nDotL = n.dot(l);
      nDotL = min(nDotL, 1.0f);

      if (nDotL > 0)
      {
         result.c.r += hitF.diffuse*hitP.c.r * nDotL * curLight->r;
         result.c.g += hitF.diffuse*hitP.c.g * nDotL * curLight->g;
         result.c.b += hitF.diffuse*hitP.c.b * nDotL * curLight->b;
      }


      // Cast light feeler ray.
      /*
         ray_t feeler;
         vec3_t dataPoint = data.point.toHost();
         feeler.dir = curLight->location - dataPoint;
         feeler.dir.normalize();
         feeler.point = feeler.dir * EPSILON;
         feeler.point += dataPoint;

         hit_t tmpHit;

      // If feeler hits any object, current point is in shadow.
      bool isShadow = gpuHit(feeler, &tmpHit);

      if (!isShadow)
      {
      // Diffuse.
      vec3_t n = hitNormal;
      n.normalize();
      vec3_t l = curLight->location - dataPoint;
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
       */
   }
   return result;
}

/*
   vec3_t Scene::reflect(vec3_t d, vec3_t n)
   {
   return n * (2 * (-d.dot(n))) + d;
   }
 */
