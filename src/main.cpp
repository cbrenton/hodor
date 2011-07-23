#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <math.h>

#include "Globals.h"
#include "Pixel.h"
#include "img/Image.h"
#include "img/TgaImage.h"
#include "img/PngImage.h"
#include "Scene.h"

#define POV_EXT ".pov"
#define DEFAULT_W 256
#define DEFAULT_H 256
#define AA_RAYS 4
#define RECURSION_DEPTH 6
#define CHUNK_SIZE 4096

using namespace std;

Image *image;
Scene *scene;
string inputFileName;
string filename;
bool useBVH = false;
int width = DEFAULT_W;
int height = DEFAULT_H;
int numAA = 1;

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
   {
      numAA = AA_RAYS;
   }
   else
   {
      numAA = atoi(strIn);
   }
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
   {
      name = strIn[1];
   }
   else
   {
      name = strIn;
   }
   inputFileName = name;
   int dirIndex = (int)inputFileName.rfind('/');
   int extIndex = (int)inputFileName.rfind(POV_EXT);
   filename = "images/";
   filename.append(inputFileName.substr(dirIndex + 1, extIndex - dirIndex - 1));
   filename.append(".png");
}

float r2d(float rads)
{
   return (float)(rads * 180 / M_PI);
}

int main(int argc, char **argv)
{
   srand((int)time(NULL));

   int c;
   while ((c = getopt(argc, argv, "a::A::bBi:I:h:H:w:W:")) != -1)
   {
      switch (c)
      {
      case 'a': case 'A':
         if (optarg != NULL)
         {
            setAA(optarg);
         }
         else
         {
            setAA((char *)"");
         }
         break;
      case 'b': case 'B':
         useBVH = true;
         break;
      case 'h': case 'H':
         setHeight(optarg);
         break;
      case 'i': case 'I':
         setFilename(optarg);
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

   image = new PngImage(width, height, filename);

   // Open input file.
   fstream inputFileStream(inputFileName.c_str(), fstream::in);

   // Check if input file is valid.
   if (inputFileStream.fail())
   {
      cerr << "File " << inputFileName << " does not exist." << endl;
      exit(EXIT_FAILURE);
   }

   // Parse scene.
   scene = Scene::read(inputFileStream);

   // Close input file.
   inputFileStream.close();

   // Make array of rays.
   /*
      ray_t ***aRayArray = new ray_t **[width];
      for (int i = 0; i < width; i++)
      {
      aRayArray[i] = new ray_t *[height];
      for (int j = 0; j < height; j++)
      {
      aRayArray[i][j] = new ray_t[numAA];
      }
      }
      */
   /*
      ray_t **aRayArray = new ray_t *[width];
      for (int i = 0; i < width; i++)
      {
      aRayArray[i] = new ray_t[height];
      }
      */
   ray_t *aRayArray = new ray_t[width * height];

   float l = -scene->camera.right.length() / 2;
   float r = scene->camera.right.length() / 2;
   float b = -scene->camera.up.length() / 2;
   float t = scene->camera.up.length() / 2;

   // Generate rays.
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
               / (float)image->width);
         float vScale = (float)(b + (t - b) * ((float)y + jitter)
               / (float)image->height);
         float wScale = -1;
         vec3_t sVector = scene->camera.location;
         vec3_t uVector = scene->camera.right;
         vec3_t vVector = scene->camera.up;
         vec3_t wVector = scene->camera.look_at - scene->camera.location;
         uVector.normalize();
         vVector.normalize();
         wVector.normalize();
         // Left-handed.
         wVector *= -1;
         uVector *= uScale;
         vVector *= vScale;
         wVector *= wScale;
         sVector += uVector;
         sVector += vVector;
         sVector += wVector;
         vec3_t rayDir = uVector + vVector + wVector;
         rayDir.normalize();
         vec3_t curPoint = vec3_t(scene->camera.location);
         //ray_t *curRay = new ray_t(curPoint, rayDir);
         ray_t curRay = {curPoint, rayDir};
         //aRayArray[i][j][k] = *curRay;
         aRayArray[x * image->height + y] = curRay;
      }
   }
   cout << "done." << endl;

   if (numAA > 1)
   {
      cout << "Using " << numAA << "x AA." << endl;
   }
   else
   {
      cout << "Antialiasing is turned off." << endl;
   }

   if (useBVH)
   {
      cout << "Using bounding volume heirarchy." << endl;
   }
   else
   {
      cout << "Not using bounding volume heirarchy." << endl;
   }

   // Initialize variables for timekeeping.
   struct timeval startTime;
   gettimeofday(&startTime, NULL);

   // Test for intersection.
   cout << "Testing intersections." << endl;

   /*
      for (int rayNdx = 0; rayNdx <= image->width * image->height / 256; rayNdx++)
      {
      cout << "starting at " << rayNdx * 256 << ", size: " << image->width * image->height << endl;
      scene->castRays(aRayArray + (rayNdx * 256), image->width, image->height,
      RECURSION_DEPTH);
      }
      */

   int numChunks = ceil((float)(width * height) / (float)CHUNK_SIZE);

   scene->cudaSetup(CHUNK_SIZE);

   Pixel *pixResults = new Pixel[width * height];
   for (int chunkNdx = 0; chunkNdx < numChunks; chunkNdx++)
   {
      Pixel *chunkPix = scene->castRays(aRayArray + chunkNdx * CHUNK_SIZE,
            min(CHUNK_SIZE, width * height - chunkNdx * CHUNK_SIZE), RECURSION_DEPTH);

      for (int pixNdx = 0; pixNdx < CHUNK_SIZE; pixNdx++)
      {
         if (pixNdx + chunkNdx * CHUNK_SIZE < width * height)
         {
            pixResults[chunkNdx * CHUNK_SIZE + pixNdx] = chunkPix[pixNdx];
         }
      }

      // Calculate and print chunk progress.
      float chunkProgress = (float)(chunkNdx + 1) / (float)numChunks * 100.0f;
      printf("\r%6.2f%%: %d/%d", chunkProgress, chunkNdx + 1, numChunks);
      fflush(stdout);
   }

   cout << endl;

   scene->cudaCleanup();

   //Pixel *pixResults = scene->castRays(aRayArray, image->width, image->height,
   //RECURSION_DEPTH);

   for (int x = 0; x < image->width; x++)
   {
      for (int y = 0; y < image->height; y++)
      {
         //Pixel curPix = scene->castRay(aRayArray[x][y], RECURSION_DEPTH);
         // Write pixel out to file.
         //image->writePixel(x, y, curPix);
         image->writePixel(x, y, pixResults[x * image->height + y]);
      }
   }

   cout << "Writing to file...";
   // finish_t writing image out to file.
   image->close();
   cout << "done." << endl;

   /*
      for (int i = 0; i < width; i++)
      {
      delete[] aRayArray[i];
      }
      */
   delete[] aRayArray;

   delete image;

   delete scene;
}
