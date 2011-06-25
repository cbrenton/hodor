#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <math.h>
// #include <cutil.h>

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

// Determines the length of the progress bar. If your terminal is being overrun, try decreasing this.
#define BAR_LEN 20

using namespace std;

Image *image;
Scene *scene;
string inputFileName;
string filename;
bool useBVH = false;
bool showProgress = true;
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

void printProgress(struct timeval startTime, int d, int total, int freq)
{
   // Initialize timekeeping variables.
   float timeLeft;
   float dt = 0;
   int seconds, useconds;
   int min, sec, ms;
   int dMin, dSec, dMs;
   min = sec = ms = dMin = dSec = dMs = 0;

   // Set padding for strings to their length (minus one for null
   // terminating character) plus a specified value.
   int strPad = 3;
   int pad = 4;

   // Get terminal width.
   struct winsize w;
   ioctl(0, TIOCGWINSZ, &w);
   int termW = w.ws_col;

   // Length of time string.
   int timeLen = 8;
   // Length of percent string.
   int percentLen = 7;

   int maxBarLen = (pad * 2 + strPad * 2) + ((int)strlen("elapsed:") - 1)
      + ((int)strlen("eta:") - 1) + (timeLen * 2) + (percentLen + 1)
      + (BAR_LEN + 2) + 1;
   int midBarLen = (pad + (int)strlen("eta:") - 1 + strPad + timeLen
         + (percentLen + 1) + 1);
   int minBarLen = percentLen + 1;

   // bool fullProgressEnabled = maxBarLen > BAR_LEN;
   bool fullProgressEnabled = maxBarLen < termW;
   bool midProgressEnabled = midBarLen < termW;
   bool minProgressEnabled = minBarLen < termW;

   if (d % freq == 0 || d == total - 1)
   {
      // Get time.
      struct timeval curTime;
      gettimeofday(&curTime, NULL);
      seconds = (int)curTime.tv_sec - (int)startTime.tv_sec;
      useconds = (int)curTime.tv_usec - (int)startTime.tv_usec;
      dt = (float)(((seconds) * 1000 + useconds/1000.0) + 0.5);
      float percent = (float)(d + 1) / (float)total;

      timeLeft = ((float)dt / percent - (float)dt) / 1000.0f;

      // Calculate time data;
      min = (int)timeLeft / 60;
      sec = (int)timeLeft % 60;
      ms = (int)(timeLeft * 100) % 60;

      dMin = (int)(dt / 1000) / 60;
      dSec = (int)(dt / 1000) % 60;
      dMs = (int)(dt / 10) % 60;

      if (fullProgressEnabled)
      {
         // Print everything.
         string progress;
         // Fill progress bar.
         progress += "[";
         for (int j = 0; j < BAR_LEN; j++)
         {
            float j_percent = (float)j / (float)BAR_LEN;
            if (j_percent < percent)
            {
               progress += "=";
            }
            else
            {
               progress += "-";
            }
         }
         progress += "]";

         // Print data.
         printf("\r%s%*s%02d:%02d:%02d",
               "elapsed:", strPad, "", dMin, dSec, dMs);
         printf("%*s%s%*s%02d:%02d:%02d",
               pad, "", "eta:", strPad, "", min, sec, ms);
         // Display progress bar.
         printf("%*s%*.2f%% %s",
               pad, "", percentLen - 2, percent * 100.0f, progress.c_str());
      }
      else if (midProgressEnabled)
      {
         // Print the percent and the ETA.
         printf("\r%-*s %02d:%02d:%02d",
               (int)strlen("eta:") - 1 + strPad, "eta:", min, sec, ms);
         printf("%*s%.2f%%",
               pad, "", percent * 100.0f);
      }
      else if (minProgressEnabled)
      {
         // Print only the percent.
         printf("\r%.2f%%", percent * 100.0f);
      }
      /*
         else
         {
         printf("Warning: terminal must be at least %d characters wide. Data will not be displayed.\n", minBarLen);
         }
         printf("terminal width: %d (%d)", termW, maxBarLen);
         */

      // Flush stdout to print stats.
      fflush(stdout);
   }
}

float r2d(float rads)
{
   return (float)(rads * 180 / M_PI);
}

int main(int argc, char **argv)
{
   srand((int)time(NULL));

   int c;
   while ((c = getopt(argc, argv, "a::A::bBi:I:h:H:pPw:W:")) != -1)
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
      Ray ***aRayArray = new Ray **[width];
      for (int i = 0; i < width; i++)
      {
      aRayArray[i] = new Ray *[height];
      for (int j = 0; j < height; j++)
      {
      aRayArray[i][j] = new Ray[numAA];
      }
      }
      */
   Ray **aRayArray = new Ray *[width];
   for (int i = 0; i < width; i++)
   {
      aRayArray[i] = new Ray[height];
   }

   float l = -scene->camera.right.norm() / 2;
   float r = scene->camera.right.norm() / 2;
   float b = -scene->camera.up.norm() / 2;
   float t = scene->camera.up.norm() / 2;

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
         Vector3f sVector = scene->camera.location;
         Vector3f uVector = scene->camera.right;
         Vector3f vVector = scene->camera.up;
         Vector3f wVector = scene->camera.look_at - scene->camera.location;
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
         Vector3f rayDir = uVector + vVector + wVector;
         rayDir.normalize();
         Vector3f curPoint = Vector3f(scene->camera.location);
         //Ray *curRay = new Ray(curPoint, rayDir);
         Ray curRay(curPoint, rayDir);
         //aRayArray[i][j][k] = *curRay;
         aRayArray[x][y] = curRay;
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
   for (int x = 0; x < image->width; x++)
   {
      for (int y = 0; y < image->height; y++)
      {
         //image->pixelData[x][y] = scene->castRay(aRayArray[x][y], RECURSION_DEPTH);
         Pixel curPix = scene->castRay(aRayArray[x][y], RECURSION_DEPTH);
         // Write pixel out to file.
         image->writePixel(x, y, curPix);
         // Print out progress bar.
         if (showProgress)
         {
            // Set the frequency of ticks to update every .01%, if possible.
            int tick = max(image->width*image->height/numAA / 10000, 100);
            printProgress(startTime, x * image->height + y,
                  image->width * image->height, tick);
         }
      }
   }
   if (showProgress)
   {
      cout << endl;
   }

   // Finish writing image out to file.
   image->close();
   
   for (int i = 0; i < width; i++)
   {
      delete[] aRayArray[i];
   }
   delete[] aRayArray;
   
   delete image;

   delete scene;
}
