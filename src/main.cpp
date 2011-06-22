#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>
#include <sstream>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <math.h>
//#include <cutil.h>

#include "Pixel.h"
#include "Image.h"
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
            //setAA(optarg);
         }
         else
         {
            //setAA((char *)"");
         }
         break;
      case 'b': case 'B':
         useBVH = true;
         break;
      case 'h': case 'H':
         //setHeight(optarg);
         break;
      case 'i': case 'I':
         //setFilename(optarg);
         break;
      case 'p': case 'P':
         showProgress = false;
         break;
      case 'w': case 'W':
         //setWidth(optarg);
         break;
      default:
         cerr << "Invalid command-line argument -" << c << endl;
         exit(EXIT_FAILURE);
         break;
      }
   }
}
