/**
 * This holds scene geometry data. Ray casting and shading take place here.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef _SCENE_H
#define _SCENE_H

#include <vector>
#include "geom/Geometry.h"
#include "Camera.h"
#include "Light.h"
#include "geom/Box.h"
#include "geom/Mesh2.h"
#include "geom/Plane.h"
#include "geom/Sphere.h"
#include "geom/Triangle.h"
#include "structs/sphere_t.h"
#include "structs/plane_t.h"
#include "structs/triangle_t.h"
#include "structs/box_t.h"
#include "structs/ray_t.h"
#include "structs/hit_t.h"
#include "Pixel.h"

#define THREADS_PER_BLOCK 256

class NYUParser;
struct hitd_t;

class Scene
{
   public:
      // Constructs a bounding volume heirarchy for the scene.
      void constructBVH();

      // Reads in scene data from a file and returns a new Scene containing the newly stored data.
      static Scene *read(std::fstream & input);

      // Checks if a ray intersects any geometry in the scene, using Geometry.
      bool cpuHit(ray_t & ray, hit_t *data);

      void cudaSetup(int chunkSize);

      void cudaCleanup();
      
      // Casts rays into the scene and returns correctly colored pixels.
      Pixel *castRays(ray_t *ray, int num, int depth);

      // Checks an array of rays for intersectionin the scene.
      hitd_t *hit(ray_t *rays, int num, int depth);

      // Casts a single ray into the scene and returns a single correctly colored pixel.
      Pixel castRay(ray_t & ray, int depth);

      // Calculates proper shading for a chunk of pixels.
      Pixel *shadeArray(hitd_t *data, ray_t *view, int num);
      
      // Calculates proper shading at the current point.
      Pixel shade(hitd_t & data, ray_t & view, bool hit);

      //vec3_t reflect(vec3_t incident, vec3_t normal);

      Camera camera;

      // List of geometry objects (CPU only).
      std::vector<Geometry*> geometry;

      // The vector of boxes in the scene (GPU only).
      std::vector<box_t*> boxes;

      // The vector of planes in the scene (GPU only).
      std::vector<plane_t*> planes;

      // The vector of spheres in the scene (GPU only).
      std::vector<sphere_t*> spheres;

      box_t *boxesArray;
      sphere_t *spheresArray;
      plane_t *planesArray;
      triangle_t *trianglesArray;
      
      box_t *boxes_d;
      sphere_t *spheres_d;
      plane_t *planes_d;
      triangle_t *triangles_d;

      size_t boxes_size;
      size_t spheres_size;
      size_t planes_size;
      size_t triangles_size;

      ray_t *rays_d;
      ray_t *rays_h;
      size_t rays_size;

      hitd_t *results_h;
      hitd_t *results_d;
      size_t results_size;

      // The vector of triangles in the scene (GPU only).
      std::vector<triangle_t*> triangles;

      // The vector of lights in the scene.
      std::vector<Light*> lights;
      
      bool useGPU;

};
#endif
