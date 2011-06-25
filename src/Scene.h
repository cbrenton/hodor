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
#include "structs//box_t.h"
#include "Ray.h"
#include "structs/HitData.h"
#include "Pixel.h"

class NYUParser;

class Scene
{
   public:
      // Constructs a bounding volume heirarchy for the scene.
      void constructBVH();

      // Reads in scene data from a file and returns a new Scene containing the newly stored data.
      static Scene* read(std::fstream & input);

      // Checks if a ray intersects any geometry in the scene.
      bool hit(const Ray & ray, HitData *data);

      // Casts a ray into the scene and returns a correctly colored pixel.
      Pixel castRay(const Ray & ray, int depth);

      // Calculates proper shading at the current point.
      Pixel shade(HitData *data, Vector3f view);

      //Vector3f reflect(Vector3f incident, Vector3f normal);

      Camera camera;

      // List of geometry objects (CPU only).
      std::vector<Geometry*> geometry;

      // The vector of spheres in the scene (GPU only).
      std::vector<sphere_t> spheres;

      // The vector of planes in the scene (GPU only).
      std::vector<plane_t> planes;

      // The vector of triangles in the scene (GPU only).
      std::vector<triangle_t> triangles;

      // The vector of boxes in the scene (GPU only).
      std::vector<box_t> boxes;

      // The vector of lights in the scene.
      std::vector<Light*> lights;

};
#endif
