#include "SFMLWindow.h"

using namespace std;
using namespace sf;

SFMLWindow::SFMLWindow(int w, int h): width(w), height(h)
{
   colorDepth = 32;
   windowTitle = "I LIKE TURTLES";

   // Set up window.
   app = new RenderWindow(VideoMode(width, height, colorDepth), windowTitle);

   app->setVisible(true);

   // Setup OpenGL.
   setupGL(width, height);
}

SFMLWindow::~SFMLWindow()
{
   // Clean up.
   app->close();
   delete app;
}

void SFMLWindow::setupGL(int width, int height)
{
   // Set color and depth clear value
   glClearDepth(1.f);
   glClearColor(0.f, 0.f, 0.f, 0.f);

   //glShadeModel(GL_FLAT);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   // Enable Z-buffer read and write
   //glEnable(GL_DEPTH_TEST);
   //glDepthMask(GL_TRUE);

   // Setup a perspective projection
   glMatrixMode(GL_PROJECTION);
   glViewport(0, 0, width, height);
   glLoadIdentity();
   gluOrtho2D(-(float)width/(float)height, (float)width/(float)height, -1., 1.);
   //gluPerspective(90.f, 1.f, 1.f, 500.f);
}

void SFMLWindow::resize(Event e)
{
   cout << "Window resized to " << app->getSize().x << " x " <<
      app->getSize().y << endl;
   glViewport(0, 0, e.size.width, e.size.height);
   glLoadIdentity();
   gluOrtho2D(-(float)e.size.width/(float)e.size.height, (float)e.size.width/(float)e.size.height, -1., 1.);
}

void SFMLWindow::update(unsigned char *image)
{
   // Clear the window.
   app->clear();

   glClear(GL_COLOR_BUFFER_BIT);
   // If an image is given, update the displayed image.
   if (image != NULL)
   {
      glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, image);
      delete image;
   }
   glFlush();
      
   // Update the main window. This is SFML.
   app->display();
}

bool SFMLWindow::isOpen()
{
   return app->isOpen();
}
