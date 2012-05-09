/**
 * Chris Brenton
 * Camera Class
 * 4/11/11
 */

#include <cstdlib>
#include <stdio.h>
#include "Camera.h"

Camera::Camera(Vector3f _loc, Vector3f _up, Vector3f _right, Vector3f _look_at) :
   location(_loc), up(_up), right(_right), look_at(_look_at)
{
}
