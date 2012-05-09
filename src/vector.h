/**
 * Vector operations.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#ifndef _VECTOR_H
#define _VECTOR_H

#include <stdio.h>
#include <math.h>

#ifdef USE_DBL
typedef double vec_t;
#else
typedef float vec_t;
#endif

struct vec2
{
   vec_t v[2];
};

struct vec3
{
   vec_t v[3];
};

// vec2 operations.
inline void debug(vec2 *a);
inline void debug(vec2 &a);

// vec3 operations.
inline vec_t distance3(vec3 *a, vec3 *b);
inline vec_t distance3(vec3 &a, vec3 &b);
inline void add(vec3 *a, vec3 *b);
inline void add(vec3 &a, vec3 &b);
inline void subtract(vec3 *a, vec3 *b);
inline void subtract(vec3 &a, vec3 &b);
inline void multiply(vec3 *a, vec_t s);
inline void multiply(vec3 &a, vec_t s);
inline vec_t dot(vec3 *a, vec3 *b);
inline vec_t dot(vec3 &a, vec3 &b);
inline vec3 cross(vec3 *a, vec3 *b);
inline vec3 cross(vec3 &a, vec3 &b);
inline void normalize(vec3 *a);
inline void normalize(vec3 &a);
inline vec_t length(vec3 *a);
inline vec_t length(vec3 &a);
inline void debug(vec3 *a);
inline void debug(vec3 &a);

// vec2 operations.
inline void debug(vec2 *a)
{
   printf("<%f, %f>\n", a->v[0], a->v[1]);
}

inline void debug(vec2 &a)
{
   printf("<%f, %f>\n", a.v[0], a.v[1]);
}

// vec3 operations.
inline vec_t distance3(vec3 *a, vec3 *b)
{
   vec_t ret0 = a->v[0] - b->v[0];
   vec_t ret1 = a->v[1] - b->v[1];
   vec_t ret2 = a->v[2] - b->v[2];

   return sqrtf(ret0 * ret0 + ret1 * ret1 + ret2 * ret2);
}

inline vec_t distance3(vec3 &a, vec3 &b)
{
   vec_t ret0 = a.v[0] - b.v[0];
   vec_t ret1 = a.v[1] - b.v[1];
   vec_t ret2 = a.v[2] - b.v[2];

   return sqrtf(ret0 * ret0 + ret1 * ret1 + ret2 * ret2);
}

inline void add(vec3 *a, vec3 *b)
{
   a->v[0] += b->v[0];
   a->v[1] += b->v[1];
   a->v[2] += b->v[2];
}

inline void add(vec3 &a, vec3 &b)
{
   a.v[0] += b.v[0];
   a.v[1] += b.v[1];
   a.v[2] += b.v[2];
}

inline void subtract(vec3 *a, vec3 *b)
{
   a->v[0] -= b->v[0];
   a->v[1] -= b->v[1];
   a->v[2] -= b->v[2];
}

inline void subtract(vec3 &a, vec3 &b)
{
   a.v[0] -= b.v[0];
   a.v[1] -= b.v[1];
   a.v[2] -= b.v[2];
}

inline void multiply(vec3 *a, vec_t s)
{
   a->v[0] *= s;
   a->v[1] *= s;
   a->v[2] *= s;
}

inline void multiply(vec3 &a, vec_t s)
{
   a.v[0] *= s;
   a.v[1] *= s;
   a.v[2] *= s;
}

inline vec_t dot(vec3 *a, vec3 *b)
{
   return a->v[0] * b->v[0] + a->v[1] * b->v[1] + a->v[2] * b->v[2];
}

inline vec_t dot(vec3 &a, vec3 &b)
{
   return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
}

inline vec3 cross(vec3 *a, vec3 *b)
{
   vec3 ret;
   ret.v[0] = a->v[1] * b->v[2] - a->v[2] * b->v[1];
   ret.v[1] = a->v[2] * b->v[0] - a->v[0] * b->v[2];
   ret.v[2] = a->v[0] * b->v[1] - a->v[1] * b->v[0];
   return ret;
}

inline vec3 cross(vec3 &a, vec3 &b)
{
   vec3 ret;
   ret.v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
   ret.v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
   ret.v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
   return ret;
}

inline void normalize(vec3 *a)
{
   vec_t len = length(a);
   if (len == 0.f)
   {
      return;
   }
   multiply(a, 1.f / len);
}

inline void normalize(vec3 &a)
{
   vec_t len = length(a);
   if (len == 0.f)
   {
      return;
   }
   multiply(a, 1.f / len);
}

inline vec_t length(vec3 *a)
{
   vec_t length = (vec_t)sqrtf((float)dot(a, a));
   return length;
}

inline vec_t length(vec3 &a)
{
   vec_t length = (vec_t)sqrtf((float)dot(a, a));
   return length;
}

inline void debug(vec3 *a)
{
   printf("<%f, %f, %f>\n", a->v[0], a->v[1], a->v[2]);
}

inline void debug(vec3 &a)
{
   printf("<%f, %f, %f>\n", a.v[0], a.v[1], a.v[2]);
}

#endif
