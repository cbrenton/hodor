
#include "Scene.h"
#include <Eigen/Dense>
using namespace Eigen;

//Constructs a bounding volume heirarchy for the scene.
void Scene::constructBVH() {
}

//Reads in scene data from a file and returns a new Scene containing the newly stored data.
Scene* Scene::read(const std::istream & input)
{
}

//Checks if a ray intersects any geometry in the scene.
bool Scene::hit(Ray ray, HitData *data) {
}

//Casts a ray into the scene and returns a correctly colored pixel.
 Scene::castRay(Ray ray, int depth) {
}

//Calculates proper shading at the current point.
Pixel Scene::shade(HitData *data, const Vector3f & view) {
}

