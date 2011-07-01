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

class NYUParser;

class Scene
{
   public:
      // Constructs a bounding volume heirarchy for the scene.
      void constructBVH();

      // Reads in scene data from a file and returns a new Scene containing the newly stored data.
      static Scene* read(std::fstream & input);

      // Checks if a ray intersects any geometry in the scene, using structs.
      bool gpuHit(ray_t & ray, hit_t *data);

      // Checks if a ray intersects any geometry in the scene, using Geometry.
      bool hit(ray_t & ray, hit_t *data);
      
      Pixel** castRays(ray_t **ray, int height, int width, int depth);

      // Casts a ray into the scene and returns a correctly color_ted pixel.
      Pixel castRay(ray_t & ray, int depth);

      // Calculates proper shading at the current point.
      Pixel shade(hit_t *data, vec3_t view);

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

      sphere_t *spheresArray;

      // The vector of triangles in the scene (GPU only).
      std::vector<triangle_t*> triangles;

      // The vector of lights in the scene.
      std::vector<Light*> lights;

      bool useGPU;

};
#endif
