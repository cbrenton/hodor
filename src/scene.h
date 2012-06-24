/**
 * This holds scene geometry data. ray casting and shading take place here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _SCENE_H
#define _SCENE_H

#include <vector>
#include "camera.h"
#include "globals.h"
#include "hit_data.h"
#include "pixel.h"
#include "ray.h"
#include "vertex.h"
#include "objLoader.h"

class Scene
{
   public:
      // Constructs a Scene from an objLoader object.
      Scene(objLoader *objScene);

      // Constructs a bounding volume heirarchy for the scene.
      void constructBVH();

      // Reads in scene data from a file and returns a new Scene containing the newly stored data.
      static Scene* read(std::string filename);

      // Checks if a ray intersects any geometry in the scene, using structs.
      bool gpuHit(ray *ray, hit_data *data);

      // Casts a ray into the scene and returns a correctly colored pixel.
      Pixel castRay(ray *ray, int depth);

      // Calculates proper shading at the current point.
      Pixel shade(hit_data *data, vec3 view);

      //vec3 reflect(vec3 incident, vec3 normal);

      Camera camera;

      // The vector of triangles in the scene.
      std::vector<triangle *> triangles;

      bool useGPU;

   protected:
      std::vector<obj_vector *> vertexList;
      std::vector<obj_vector *> textureList;
      std::vector<obj_vector *> normalList;
};
#endif
