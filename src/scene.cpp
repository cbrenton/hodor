/**
 * This holds scene geometry data. ray casting and shading take place here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#include "scene.h"

using namespace glm;

/**
 * Constructs a Scene from an objLoader object.
 */
Scene::Scene(objLoader *objScene)
{
   vertexList.assign(objScene->vertexList,
         objScene->vertexList + objScene->vertexCount);
   normalList.assign(objScene->normalList,
         objScene->normalList + objScene->normalCount);
   textureList.assign(objScene->textureList,
         objScene->textureList + objScene->textureCount);
   for (int i = 0; i < objScene->faceCount; i++)
   {
      if (objScene->faceList[i]->vertex_count == 4)
      {
         triangle *quadTris = quadToTri(objScene->faceList[i], objScene);
         triangles.push_back(quadTris);
         triangles.push_back(quadTris + 1);
      }
      else
      {
         triangles.push_back(faceToTri(objScene->faceList[i], objScene));
      }
   }
}

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
Scene* Scene::read(std::string filename)
{
   // Parse an obj file.
   objLoader *objData = new objLoader();
   objData->load((char *)filename.c_str());

   // Convert data to a usable format.
   Scene *curScene = new Scene(objData);
   
   // TODO: Actually parse the camera/handle cases where no camera is specified.
   glm::vec3 location(0.f, 0.f, -10.f);
   glm::vec3 up(0.f, 1.f, 0.f);
   glm::vec3 right(1.f, 0.f, 0.f);
   glm::vec3 look_at(0.f, 0.f, 0.f);
   curScene->camera = Camera();
   curScene->camera.location = location;
   curScene->camera.up = up;
   curScene->camera.right = right;
   glm::vec3 d = look_at - location;
   d = normalize(d);
   curScene->camera.look_at = look_at;

   delete objData;
   return curScene;
}

/**
 * Checks if a ray intersects any geometry in the scene, using structs.
 * @returns true if an intersection is found.
 */
bool Scene::gpuHit(ray *ray, hit_data *data)
{
   //debug(ray);
   // INITIALIZE closestT to MAX_DIST + 0.1
   float closestT = MAX_DIST + 0.1f;
   // INITIALIZE closestData to empty hit_data
   hit_data *closestData = new hit_data;

   // Find hit for triangles.
   // FOR each item in triangles
   // TODO: Make this work.
   for (int triNdx = 0; triNdx < (int)triangles.size(); triNdx++)
   {
      float triT = -1;
      hit_data *triData = new hit_data;
      // IF current item is hit by ray
      /*
      debug(triangles[triNdx]->pts[0]);
      debug(triangles[triNdx]->pts[1]);
      debug(triangles[triNdx]->pts[2]);
      */
      if (hit(ray, triangles[triNdx], &triT, triData))
      {
         //printf("hit!");
         // IF intersection is closer than closestT
         if (triT < closestT)
         {
            //printf(" closer than before!");
            // SET closestT to intersection
            closestT = triT;
            // SET closestData to intersection data
            *closestData = *triData;
            closestData->objIndex = triNdx;
         }
         //printf("\n");
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
   //printf("%f ?= %f\n", closestT, MAX_DIST);
   return (closestT <= MAX_DIST);
}

// Casts a ray into the scene and returns a correctly colored pixel.
vec3 Scene::castRay(ray *ray, int depth)
{
   vec3 result(0.f, 0.f, 0.f);
   vec3 reflectPix(0.f, 0.f, 0.f);
   vec3 refractPix(0.f, 0.f, 0.f);
   hit_data rayData;
   //if (hit(ray, &rayData))
   if (gpuHit(ray, &rayData))
   {
      result = shade(&rayData, ray->dir);
      if (useGPU)
      {
         // TODO: Add reflect to hit_data.
         //if (rayData.reflect != NULL && depth > 0)
         if (false)
         {
            // TODO: Make this work with triangle.
            /*
               Pigment hitP = {};
               Finish hitF = {};
               vec3 hitNormal(0.0, 0.0, 0.0);
               triangle tri;
               tri = triangles[rayData.objIndex];
               hitP = tri.p;
               hitF = tri.f;
               hitNormal = triangle_normal(tri);

               vec3 reflectVec = *rayData.reflect;
               reflectVec.normalize();
               ray reflectray(rayData.point + reflectVec * EPSILON, reflectVec);

               reflectPix = castray(reflectray, depth - 1);

               reflectPix.multiply(hitF.reflection);
               result.multiply(1 - hitF.reflection);
               result.add(reflectPix);
               */
         }
      }
   }
   return result;
}

// Calculates proper shading at the current point.
vec3 Scene::shade(hit_data *data, vec3 view)
{
   vec3 result(1.f, 0.f, 0.f);
   //printf("hit\n");
   // TODO: Make this work with EVERYTHING.
   /*
      Pigment hitP = {};
      Finish hitF = {0};
      vec3 hitNormal(0.0, 0.0, 0.0);
      triangle_t tri;
      tri = triangles[data->objIndex];
      hitP = tri.p;
      hitF = tri.f;
      hitNormal = triangle_normal(tri);

      for (int lightNdx = 0; lightNdx < (int)lights.size(); lightNdx++)
      {
      Light *curLight = lights[lightNdx];
   // Ambient.
   // GPU.
   if (useGPU)
   {
   result.c.r += (hitF.ambient*hitP.c.r) * curLight->r;
   result.c.g += (hitF.ambient*hitP.c.g) * curLight->g;
   result.c.b += (hitF.ambient*hitP.c.b) * curLight->b;
   }
   // CPU.
   else
   {
   result.c.r += (data->object->f.ambient*data->object->p.c.r) *
   curLight->r;
   result.c.g += (data->object->f.ambient*data->object->p.c.g) *
   curLight->g;
   result.c.b += (data->object->f.ambient*data->object->p.c.b) *
   curLight->b;
   }

   // Cast light feeler ray.
   ray feeler;
   feeler.dir = curLight->location - data->point;
   float lightLen = feeler.dir.norm();
   feeler.dir.normalize();
   feeler.point = data->point + feeler.dir * EPSILON;

   hit_data tmpHit;

   // If feeler hits any object, current point is in shadow.
   bool isShadow = gpuHit(feeler, &tmpHit);

   if (!isShadow || tmpHit.t > lightLen)
   {
   // GPU.
   if (useGPU)
   {
   // Diffuse.
   vec3 n = hitNormal;
   n.normalize();
   vec3 l = curLight->location - data->point;
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
   vec3 r = mReflect(l, n);
   r.normalize();
   vec3 v = view;
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
// CPU.
else
{
   // Diffuse.
   vec3 n = data->object->getNormal(data->point);
   n.normalize();
   vec3 l = curLight->location - data->point;
   l.normalize();
   float nDotL = n.dot(l);
   nDotL = min(nDotL, 1.0f);
   if (nDotL < 0)
   {
      nDotL *= -1;
   }

   if (nDotL > 0)
   {
      result.c.r += data->object->f.diffuse*data->object->p.c.r *
         nDotL * curLight->r;
      result.c.g += data->object->f.diffuse*data->object->p.c.g *
         nDotL * curLight->g;
      result.c.b += data->object->f.diffuse*data->object->p.c.b *
         nDotL * curLight->b;
   }

   // Specular (Phong).
   vec3 r = mReflect(l, n);
   r.normalize();
   vec3 v = view;
   v.normalize();
   float rDotV = r.dot(v);
   rDotV = (float)pow(rDotV, 1.0f / data->object->f.roughness);
   rDotV = min(rDotV, 1.0f);

   if (rDotV > 0)
   {
      result.c.r += data->object->f.specular*data->object->p.c.r *
         rDotV * curLight->r;
      result.c.g += data->object->f.specular*data->object->p.c.g *
         rDotV * curLight->g;
      result.c.b += data->object->f.specular*data->object->p.c.b *
         rDotV * curLight->b;
   }
}
}
}
*/
return result;
}

/*
   vec3 Scene::reflect(vec3 d, vec3 n)
   {
   return n * (2 * (-d.dot(n))) + d;
   }
   */
