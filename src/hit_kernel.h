/**
 * Contains intersection data for all geometry structs.
 * @author Chris Brenton
 * @date 6/24/2011
 */

#include "structs/cuda_vector.h"

struct box_t;
struct plane_t;
struct sphere_t;
struct triangle_t;
struct hitd_t;
struct color_t;
struct ray_t;
struct hit_t;

__device__ box_t *boxes;
__device__ plane_t *planes;
__device__ sphere_t *spheres;
__device__ triangle_t *triangles;

__device__ int boxes_size = 0;
__device__ int planes_size = 0;
__device__ int spheres_size = 0;
__device__ int triangles_size = 0;

int cpu_hit(box_t *b_t, ray_t & ray, float *t, hitd_t *data);
int cpu_hit(plane_t *p_t, ray_t & ray, float *t, hitd_t *data);
int cpu_hit(sphere_t & s_t, ray_t & ray, float *t, hit_t *data);
int cpu_hit(triangle_t *t_t, ray_t & ray, float *t, hitd_t *data);

__device__ int box_hit(box_t & b_t, ray_t & ray, float *t, hitd_t *data);
__device__ int plane_hit(plane_t & p_t, ray_t & ray, float *t, hitd_t *data);
__device__ int sphere_hit(sphere_t & s_t, ray_t & ray, float *t, hitd_t *data);
__device__ int triangle_hit(triangle_t & t_t, ray_t & ray, float *t, hitd_t *data);

vec3_t normal(box_t & b_t, hitd_t & data);
vec3_t normal(plane_t & p_t);
vec3_t normal(sphere_t & s_t, vec3_t & data);
vec3_t normal(triangle_t & t_t);

__global__ void set_boxes(box_t *boxesIn, int numBoxes);
__global__ void set_spheres(sphere_t *spheresIn, int numSpheres);
__global__ void set_planes(plane_t *planesIn, int numPlanes);
__global__ void set_triangles(triangle_t *trianglesIn, int numTriangles);

__global__ void cuda_hit(ray_t *rays, int num, hitd_t *results);
//__global__ void cuda_hit(ray_t *rays, int num, sphere_t *spheres,
      //int spheres_size, hitd_t *results);
