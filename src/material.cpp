/**
 * Material struct operations.
 * @author Chris Brenton
 * @date 2012/05/08
 */
#include "material.h"

using namespace std;

material * matFromFile(string filename)
{
   material *ret = new material;
   ret->name = filename;
   ret->albedo = new buffer3;
   //initBuffer3(ret->albedo, 100, 100);
   drawStripes(ret->albedo, 1000, 1000);
   return ret;
}

void debug(material *mat)
{
   printf("material:\n\t%s\n", mat->name.c_str());
   printf("\t[0][0]: ");
   debug(lookup(mat->albedo, 0, 0));
}
