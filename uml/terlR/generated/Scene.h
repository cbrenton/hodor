#ifndef _SCENE_H
#define _SCENE_H


#include "Geometry.h"
#include "sphere_t.h"
#include "plane_t.h"
#include "triangle_t.h"
#include "box_t.h"
#include "Ray.h"
#include "HitData.h"
#include "Pixel.h"

class Scene
{
  public:
    //List of geometry objects (CPU only).
    Geometry geometry;


  protected:
    //The vector of spheres in the scene (GPU only).
    sphere_t spheres;

    //The vector of planes in the scene (GPU only).
    plane_t planes;

    //The vector of triangles in the scene (GPU only).
    triangle_t triangles;

    //The vector of boxes in the scene (GPU only).
    box_t boxes;


  public:
    //Constructs a bounding volume heirarchy for the scene.
    void constructBVH();

    //Reads in scene data from a file and returns a new Scene containing the newly stored data.
    static Scene* read(std::istream & input);

    //Checks if a ray intersects any geometry in the scene.
    bool hit(Ray ray, HitData *data);

    //Casts a ray into the scene and returns a correctly colored pixel.
     castRay(Ray ray, int depth);

    //Calculates proper shading at the current point.
    Pixel shade(HitData *data, const Vector3f & view);

};
#endif
