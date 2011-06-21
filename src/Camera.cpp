/**
 * Chris Brenton
 * Camera Class
 * 4/11/11
 */

#include <cstdlib>
#include <stdio.h>
#include "Camera.h"

#define EXP_ARGS 12

Camera::Camera(std::istream& input) {
   std::string line;
   getline(input, line);
   int scan = 0;
   // If the line is only an opening curly brace, skip it.
   if (line == "{")
   {
      // Get the good stuff.
      getline(input, line);
      scan += sscanf(line.c_str(), " location <%f, %f, %f>",
            &location(0), &location(1), &location(2));
      getline(input, line);
      scan += sscanf(line.c_str(), " up <%f, %f, %f>",
            &up(0), &up(1), &up(2));
      getline(input, line);
      scan += sscanf(line.c_str(), " right <%f, %f, %f>",
            &right(0), &right(1), &right(2));
      getline(input, line);
      scan += sscanf(line.c_str(), " look_at <%f, %f, %f>",
            &look_at(0), &look_at(1), &look_at(2));
   }
   else
   {
      scan += sscanf(line.c_str(), " { location <%f, %f, %f>",
            &location(0), &location(1), &location(2));
      getline(input, line);
      scan += sscanf(line.c_str(), " up <%f, %f, %f>",
            &up(0), &up(1), &up(2));
      getline(input, line);
      scan += sscanf(line.c_str(), " right <%f, %f, %f>",
            &right(0), &right(1), &right(2));
      getline(input, line);
      scan += sscanf(line.c_str(), " look_at <%f, %f, %f>",
            &look_at(0), &look_at(1), &look_at(2));
   }
   if (scan < EXP_ARGS)
   {
      printf("Invalid camera format: expected %d, found %d.\n", scan, EXP_ARGS);
      std::cout << "\tline: " << line << std::endl;
      exit(EXIT_FAILURE);
   }
}
