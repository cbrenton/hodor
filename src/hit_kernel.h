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
struct hit_t;
struct hitd_t;
struct color_t;
struct ray_t;

int box_hit(box_t *b_t, ray_t & ray, float *t, hit_t *data);

vec3d_t box_normal(box_t *b_t, hit_t *data);

int plane_hit(plane_t *p_t, ray_t & ray, float *t, hit_t *data);

vec3d_t plane_normal(plane_t *p_t);

__device__ int sphere_hit(sphere_t & s_t, ray_t & ray, float *t, hitd_t *data);

vec3d_t sphere_normal(sphere_t & s_t, hit_t *data);

int triangle_hit(triangle_t *t_t, ray_t & ray, float *t, hit_t *data);

vec3d_t triangle_normal(triangle_t *t_t);

__global__ void hit_spheres(ray_t *rays, int width, int height,
      sphere_t *spheres, int spheres_size, hitd_t *results);
