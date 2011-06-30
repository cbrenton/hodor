/**
 * Chris Brenton
 * Camera Class
 * 4/11/11
 */

#include <cstdlib>
#include <stdio.h>
#include "Camera.h"

Camera::Camera(vec3_t _loc, vec3_t _up, vec3_t _right, vec3_t _look_at) :
   location(_loc), up(_up), right(_right), look_at(_look_at)
{
}
