#include <iostream>
#include <obj.hpp>
#include <string>
#include "buffer.h"
#include "material.h"
#include "ray.h"
#include "vector.h"
#include "vertex.h"

using namespace std;

void vertex_callback(obj::float_type x, obj::float_type y, obj::float_type z);

int main(int argc, char **argv)
{
   obj::obj_parser parse;
   parse.geometric_vertex_callback(vertex_callback);
   parse.parse("input/cube.obj");
   material *m = new material;
   string name = "test.png";
   m = matFromFile(name);
   debug(m);
   printToFile(m->albedo, name);
}

void vertex_callback(obj::float_type x, obj::float_type y, obj::float_type z)
{
   vertex vert;
   vert.coord.v[0] = (vec_t)x;
   vert.coord.v[1] = (vec_t)y;
   vert.coord.v[2] = (vec_t)z;
   printf("vert: ");
   debug(&vert);
}
