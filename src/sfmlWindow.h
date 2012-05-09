/**
 * SFMLWindow class. Displays an image in an SFML window.
 * @author Chris Brenton
 * @date 2012-03-29
 */

#ifndef _SFMLWINDOW_H
#define _SFMLWINDOW_H

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

class SFMLWindow {
   public:
      SFMLWindow(int w, int h);
      ~SFMLWindow();
      
      void initialize();
      void setupGL(int width, int height);
      void resize(sf::Event::SizeEvent e);
      void update(unsigned char *image = NULL);
      bool isOpen();

   private:
      int width, height, colorDepth;
      std::string windowTitle;
      sf::RenderWindow *app;
};

#endif
