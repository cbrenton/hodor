#include <cstdlib>
#include <cstring>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <sys/ioctl.h>

#include "buffer.h"
#include "globals.h"
#include "material.h"
#include "pixel.h"
#include "ray.h"
#include "scene.h"
#include "sfmlWindow.h"
#include "glm/glm.hpp"
#include "vertex.h"
#include "img/image.h"

#define AA_RAYS 4
#define DEFAULT_H 256
#define DEFAULT_W 256
#define INPUT_EXT ".obj"
#define RECURSION_DEPTH 6

using namespace std;

Image *image;
Scene *scene;
SFMLWindow *win;
string inputFileName;
string filename;
bool useBVH = false;
bool useGPU = false;
bool showPreview = false;
bool showProgress = true;
int width = DEFAULT_W;
int height = DEFAULT_H;
int numAA = 1;

void setWidth(char* strIn);
void setHeight(char* strIn);
void setAA(char* strIn);
void setFilename(char* strIn);

int main(int argc, char **argv)
{
   srand((int)time(NULL));

   int c;
   while ((c = getopt(argc, argv, "a::A::bBgGi:I:h:H:pPw:W:")) != -1)
   {
      switch (c)
      {
      case 'a': case 'A':
         if (optarg != NULL)
            setAA(optarg);
         else
            setAA((char *)"");
         break;
      case 'b': case 'B':
         useBVH = true;
         break;
      case 'g': case 'G':
         useGPU = true;
         break;
      case 'h': case 'H':
         setHeight(optarg);
         break;
      case 'i': case 'I':
         setFilename(optarg);
         break;
      case 'p': case 'P':
         showProgress = false;
         break;
      case 'w': case 'W':
         setWidth(optarg);
         break;
      default:
         cerr << "Invalid command-line argument -" << c << endl;
         exit(EXIT_FAILURE);
         break;
      }
   }

   if (showPreview)
   {
      win = new SFMLWindow(width, height);
      win->update();
   }

   image = new Image(width, height, filename);

   // Parse scene.
   scene = Scene::read(inputFileName);
   scene->useGPU = useGPU;

   // Make array of rays.
   // TODO: Add AA.
   ray **aRayArray = new ray *[width];
   for (int i = 0; i < width; i++)
   {
      aRayArray[i] = new ray[height];
   }

   /*
   float l = length(negative(scene->camera.right)) / 2.f;
   float r = length(scene->camera.right) / 2.f;
   float b = length(negative(scene->camera.up)) / 2.f;
   float t = length(scene->camera.up) / 2.f;
   */
   float l = length(-1.0f * (scene->camera.right));
   float r = length(scene->camera.right);
   float b = length(-1.0f * (scene->camera.up));
   float t = length(scene->camera.up);

   /*
   printf("l: %f\n", l);
   printf("r: %f\n", r);
   printf("b: %f\n", b);
   printf("t: %f\n", t);
   */

   // Generate rays.
   // TODO: Add AA.
   cout << "Generating rays...";
   for (int x = 0; x < image->width; x++)
   {
      for (int y = 0; y < image->height; y++)
      {
         float jitter = 0.5f;
         if (numAA > 1)
         {
            //jitter = randFloat();
         }

         float uScale = (float)(l + (r - l) * ((float)x + jitter)
               / (float)width);
         float vScale = (float)(b + (t - b) * ((float)y + jitter)
               / (float)height);
         float wScale = -1;
         vec3 sVector = scene->camera.location;
         vec3 uVector = scene->camera.right;
         vec3 vVector = scene->camera.up;
         vec3 wVector = scene->camera.look_at - scene->camera.location;
         normalize(uVector);
         normalize(vVector);
         normalize(wVector);
         // Left-handed.
         wVector *= -1;
         uVector *= uScale;
         vVector *= vScale;
         wVector *= wScale;
         sVector += uVector;
         sVector += vVector;
         sVector += wVector;
         vec3 rayDir = uVector + vVector + wVector;
         rayDir = normalize(rayDir);
         vec3 curPoint = scene->camera.location;


         /*
         float uScale = (float)(l + (r - l) * ((float)x + jitter)
               / (float)image->width);
         float vScale = (float)(b + (t - b) * ((float)y + jitter)
               / (float)image->height);
         float wScale = -1;
         glm::vec3 sVector = scene->camera.location;
         glm::vec3 uVector = normalize(scene->camera.right);
         glm::vec3 vVector = normalize(scene->camera.up);
         glm::vec3 wVector = normalize(scene->camera.look_at - scene->camera.location);
         // Left-handed.
         wVector *= -1.0f;
         uVector *= uScale;
         vVector *= vScale;
         wVector *= wScale;
         sVector += uVector;
         sVector += vVector;
         sVector += wVector;
         glm::vec3 rayDir = uVector;
         rayDir += vVector;
         rayDir += wVector;
         rayDir = normalize(rayDir);
         glm::vec3 curPoint = scene->camera.location;
         */
         //ray *curRay = new ray(curPoint, rayDir);
         ray curRay = {curPoint, rayDir};
         //aRayArray[i][j][k] = *curRay;
         aRayArray[x][y] = curRay;
      }
   }
   cout << "done." << endl;

   if (numAA > 1)
      cout << "Using " << numAA << "x AA." << endl;
   else
      cout << "Antialiasing is turned off." << endl;

   if (useBVH)
      cout << "Using bounding volume heirarchy." << endl;
   else
      cout << "Not using bounding volume heirarchy." << endl;

   // Initialize variables for timekeeping.
   struct timeval startTime;
   gettimeofday(&startTime, NULL);

   // Test for intersection.
   cout << "Testing intersections." << endl;
   for (int x = 0; x < image->width; x++)
   {
      for (int y = 0; y < image->height; y++)
      {
         // TODO: Replace Pixel.
         Pixel curPix = scene->castRay(&aRayArray[x][y], RECURSION_DEPTH);
         //if (curPix.c.r > 0.0f)
            //printf("pix: <%f, %f, %f>\n", curPix.c.r, curPix.c.g, curPix.c.b);
         // Write pixel out to file.
         // TODO: Write correctly.
         //image->writePixel(x, y, curPix);
         glm::vec3 color = glm::vec3(curPix.c.r, curPix.c.g, curPix.c.b);
         image->setPixel(x, y, &color);
         // Print out progress bar.
         //if (showProgress)
         // TODO: Display progress bar.
      }
      //if (showPreview)
         //win->update(image->getPixelBuffer());
   }
   if (showProgress)
      cout << endl;

   // Finish writing image out to file.
   image->write();

   for (int i = 0; i < width; i++)
   {
      delete[] aRayArray[i];
   }
   delete[] aRayArray;

   delete image;

   delete scene;

   if (showPreview)
      delete win;

   return EXIT_SUCCESS;
}

void setWidth(char* strIn)
{
   width = atoi(strIn);
   if (width <= 0)
   {
      cerr << "Invalid width.\n";
      exit(EXIT_FAILURE);
   }
}

void setHeight(char* strIn)
{
   height = atoi(strIn);
   if (height <= 0)
   {
      cerr << "Invalid height: " << height << endl;
      exit(EXIT_FAILURE);
   }
}

void setAA(char* strIn)
{
   if (strlen(strIn) == 0)
      numAA = AA_RAYS;
   else
      numAA = atoi(strIn);
   if (numAA < 1)
   {
      cerr << "Invalid antialiasing sample rate: " << numAA << endl;
      exit(EXIT_FAILURE);
   }
}

void setFilename(char* strIn)
{
   string name = "";
   if (strIn[0] == '=')
      name = strIn[1];
   else
      name = strIn;
   inputFileName = name;
   int dirIndex = (int)inputFileName.rfind('/');
   int extIndex = (int)inputFileName.rfind(INPUT_EXT);
   filename = "images/";
   filename.append(inputFileName.substr(dirIndex + 1, extIndex - dirIndex - 1));
   filename.append(".tga");
}

float r2d(float rads)
{
   return (float)(rads * 180 / M_PI);
}

