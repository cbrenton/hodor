/**
 * Basic vector struct.
 * @author Chris Brenton
 * @date 07/01/2011
 */

#ifndef _CUDA_VECTOR_H
#define _CUDA_VECTOR_H

#include "structs/vector.h"

struct vec3d_t
{
   __device__ vec3d_t() {v[0] = v[1] = v[2] = 0;}
   __device__ vec3d_t(float px, float py, float pz) {v[0] = px; v[1] = py; v[2] = pz;}
   __device__ vec3d_t(vec3d_t &pVec) {v[0] = pVec.v[0]; v[1] = pVec.v[1]; v[2] = pVec.v[2];}
   __device__ vec3d_t(vec3_t &pVec) {v[0] = pVec.v[0]; v[1] = pVec.v[1]; v[2] = pVec.v[2];}
   __device__ vec3d_t(float *pVec) {v[0] = pVec[0]; v[1] = pVec[1]; v[2] = pVec[2];}

   __device__ vec3d_t operator=(vec3d_t &pVec)
   __device__ {return vec3d_t(v[0] = pVec.v[0], v[1] = pVec.v[1], v[2] = pVec.v[2]);}
   __device__ vec3d_t operator=(float *ptr)
   __device__ {return vec3d_t(v[0] = ptr[0], v[1] = ptr[1], v[2] = ptr[2]);}
   __device__ int operator==(vec3d_t &pVec)
   __device__ {return (v[0] == pVec.v[0] && v[1] == pVec.v[1] && v[2] == pVec.v[2]);}
   __device__ int operator==(float *pVec)
   __device__ {return (v[0] == pVec[0] && v[1] == pVec[1] && v[2] == pVec[2]);}
   __device__ inline int operator!=(vec3d_t &pVec)
   __device__ {return !(pVec == (*this));}
   __device__ inline int operator!=(float *pVec)
   __device__ {return !(pVec == (*this));}

   __device__ vec3d_t operator+=(vec3d_t &pVec);
   __device__ vec3d_t operator-=(vec3d_t &pVec);
   __device__ vec3d_t operator*=(vec3d_t &pVec);
   __device__ vec3d_t operator*=(float val);
   __device__ vec3d_t operator/=(vec3d_t &pVec);
   __device__ vec3d_t operator/=(float val);

   __device__ vec3d_t operator+(vec3d_t &pVec)
   __device__ {return vec3d_t(v[0] + pVec.v[0], v[1] + pVec.v[1], v[2] + pVec.v[2]);}
   __device__ vec3d_t operator-(vec3d_t &pVec)
   __device__ {return vec3d_t(v[0] - pVec.v[0], v[1] - pVec.v[1], v[2] - pVec.v[2]);}
   __device__ vec3d_t operator*(vec3d_t &pVec)
   __device__ {return vec3d_t(v[0] * pVec.v[0], v[1] * pVec.v[1], v[2] * pVec.v[2]);}
   __device__ vec3d_t operator*(float val)
   __device__ {return vec3d_t(v[0] * val, v[1] * val, v[2] * val);}
   __device__ vec3d_t operator/(vec3d_t &pVec)
   __device__ {return vec3d_t(v[0] / pVec.v[0], v[1] / pVec.v[1], v[2] / pVec.v[2]);}
   __device__ vec3d_t operator/(float val)
   __device__ {return vec3d_t(v[0] / val, v[1] / val, v[2] / val);}

   __device__ void clear(void) {v[0] = v[1] = v[2] = 0;}
   __device__ void normalize(void);
   __device__ float length(void);
   __device__ float dot(vec3d_t &pVec) {return v[0] * pVec.v[0] + v[1] * pVec.v[1] + v[2] * pVec.v[2];}
   __device__ void cross(vec3d_t &p, vec3d_t &q);

   __device__ void set(float x, float y, float z) {v[0] = x; v[1] = y; v[2] = z;}

   __device__ float x(void) {return v[0];}
   __device__ float y(void) {return v[1];}
   __device__ float z(void) {return v[2];}
   __device__ void x(float nx) {v[0] = nx;}
   __device__ void y(float ny) {v[1] = ny;}
   __device__ void z(float nz) {v[2] = nz;}

   __device__ const float &operator[](int ndx) const {return v[ndx];}
   __device__ float &operator[](int ndx) {return v[ndx];}
   __device__ operator float*(void) {return v;}

   __device__ void clamp(int min, int max);

   __device__ void rotateX(float amnt);
   __device__ void rotateY(float amnt);
   __device__ void rotateZ(float amnt);

   __host__ inline vec3_t toHost()
   {
      return vec3_t(v);
   }

   __device__ void print();

   //protected:
   float v[3];
};

__device__ inline vec3d_t vec3d_t::operator+=(vec3d_t &pVec)
{
   vec3d_t ret;

   ret = *this = *this + pVec;

   return ret;
}

__device__ inline vec3d_t vec3d_t::operator-=(vec3d_t &pVec)
{
   vec3d_t ret;

   ret = *this = *this - pVec;

   return ret;
}

__device__ inline vec3d_t vec3d_t::operator*=(vec3d_t &pVec)
{
   vec3d_t ret;

   ret = *this = *this * pVec;

   return ret;
}

__device__ inline vec3d_t vec3d_t::operator*=(float val)
{
   vec3d_t ret;

   ret = *this = *this * val;

   return ret;
}

__device__ inline vec3d_t vec3d_t::operator/=(vec3d_t &pVec)
{
   vec3d_t ret;

   ret = *this = *this / pVec;

   return ret;
}

__device__ inline vec3d_t vec3d_t::operator/=(float val)
{
   vec3d_t ret;

   ret = *this = *this / val;

   return ret;
}

__device__ inline void vec3d_t::normalize(void)
{
   float len0, len1 = 0;

   len0 = length();

   if (len0 == 0)
      return;

   len1 = 1.0f / len0;

   v[0] *= len1;
   v[1] *= len1;
   v[2] *= len1;
}

__device__ inline float vec3d_t::length(void)
{
   double length = 0;

   length = (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]);

   return (float) sqrt(length);
}

__device__ inline void vec3d_t::cross(vec3d_t &p, vec3d_t &q)
{
   v[0] = p.v[1] * q.v[2] - p.v[2] * q.v[1];
   v[1] = p.v[2] * q.v[0] - p.v[0] * q.v[2];
   v[2] = p.v[0] * q.v[1] - p.v[1] * q.v[0];
}

__device__ inline void vec3d_t::clamp(int min, int max)
{
   if (v[0] > max || v[0] < min)
      v[0] = 0;

   if (v[1] > max || v[1] < min)
      v[1] = 0;

   if (v[2] > max || v[2] < min)
      v[2] = 0;
}

__device__ inline void vec3d_t::rotateX(float amnt)
{
   float s = (float)sin(amnt);
   float c = (float)cos(amnt);
   float y = v[1];
   float z = v[2];

   v[1] = (y * c) - (z * s);
   v[2] = (y * s) + (z * c);
}

__device__ inline void vec3d_t::rotateY(float amnt)
{
   float s = (float)sin(amnt);
   float c = (float)cos(amnt);
   float x = v[0];
   float z = v[2];

   v[0] = (x * c) + (z * s);
   v[2] = (z * c) - (x * s);
}

__device__ inline void vec3d_t::rotateZ(float amnt)
{
   float s = (float)sin(amnt);
   float c = (float)cos(amnt);
   float x = v[0];
   float y = v[1];

   v[0] = (x * c) - (y * s);
   v[1] = (y * c) + (x * s);
}

#endif
