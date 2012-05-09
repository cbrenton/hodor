/**
 * Chris Brenton
 * Camera Class
 * 4/11/11
 */

#include <cstdlib>
#include <stdio.h>
#include "camera.h"

Camera::Camera(vec3 _loc, vec3 _up, vec3 _right, vec3 _look_at) :
   location(_loc), up(_up), right(_right), look_at(_look_at)
{
}
