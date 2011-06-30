/**
 * Contains intersection data for all geometry structs.
 * @author Chris Brenton
 * @date 6/24/2011
 */

#include <structs/vector.h>

struct box_t;
struct plane_t;
struct sphere_t;
struct triangle_t;
struct hit_t;
struct color_t;
class ray_t;

int box_hit(box_t *b_t, ray_t & ray, float *t, hit_t *data);

vec3_t box_normal(box_t *b_t, hit_t *data);

int plane_hit(plane_t *p_t, ray_t & ray, float *t, hit_t *data);

vec3_t plane_normal(plane_t *p_t);

int sphere_hit(sphere_t *s_t, ray_t & ray, float *t, hit_t *data);

vec3_t sphere_normal(sphere_t *s_t, hit_t *data);

int triangle_hit(triangle_t *t_t, ray_t & ray, float *t, hit_t *data);

vec3_t triangle_normal(triangle_t *t_t);

//__global__ void hitSpheres(sphere_t *sData, 
