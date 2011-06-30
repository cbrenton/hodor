/**
 * Contains intersection data for all geometry structs.
 * @author Chris Brenton
 * @date 6/24/2011
 */

struct box_t;
struct plane_t;
struct sphere_t;
struct triangle_t;
struct HitData;
class Ray;

#include <structs/vector.h>

int box_hit(box_t *b_t, Ray & ray, float *t, HitData *data);

vec3_t box_normal(box_t *b_t, HitData *data);

int plane_hit(plane_t *p_t, Ray & ray, float *t, HitData *data);

vec3_t plane_normal(plane_t *p_t);

int sphere_hit(sphere_t *s_t, Ray & ray, float *t, HitData *data);

vec3_t sphere_normal(sphere_t *s_t, HitData *data);

int triangle_hit(triangle_t *t_t, Ray & ray, float *t, HitData *data);

vec3_t triangle_normal(triangle_t *t_t);
