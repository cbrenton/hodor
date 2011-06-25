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

#include <Eigen/Dense>

using Eigen::Vector3f;

int box_hit(const box_t & b_t, const Ray & ray, float *t, HitData *data);

Vector3f boxNormal(const box_t & b_t, HitData *data);

int plane_hit(const plane_t & p_t, const Ray & ray, float *t, HitData *data);

Vector3f planeNormal(const plane_t & p_t);

int sphere_hit(const sphere_t & s_t, const Ray & ray, float *t, HitData *data);

Vector3f sphereNormal(const sphere_t & s_t, HitData *data);

int triangle_hit(const triangle_t & t_t, const Ray & ray, float *t, HitData *data);

Vector3f triangleNormal(const triangle_t & t_t);
